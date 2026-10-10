# PE-3k / PE-4k + Qwen-PE → native 2K DiT → UTP 4×256 GSB

本目录记录训练 run `39c59fafbc35` 的完整流程。`pe3k` 对应 checkpoint-3000，`pe4k` 对应 checkpoint-4000。两者各有独立模型锁和实际执行源码包；旧版 PE-29k / DiT44k 入口仍在仓库根目录。

**PE-3k 入口：** [PE3K.md](PE3K.md)，模型配置为 [profiles/pe3k.json](profiles/pe3k.json)。下方命令默认使用 PE-3k。切换 PE-4k 时将 `PE_PROFILE` 改为 `pe4k`，并使用新的 workspace 和 S3 前缀。

## 1. 固定模型与推理协议

| 阶段 | 固定配置 |
|---|---|
| 输入 | 1,464 条冻结样本；source、原始指令和目标画布身份校验 |
| Qwen-PE | Qwen/Qwen-Image-2.1-PE-I2I，revision `72927bc08afc99b7888ceb7d7d51a12db3700bbd`；复用已验证 `rewritten_prompt` |
| PE-3k | run `39c59fafbc35`，checkpoint-3000，trained-MTP HF export |
| PE-4k | 同一 run，checkpoint-4000，独立 trained-MTP HF export |
| PE prompt | `luna_source_target_size_ordered_sparse_reference_edits_v1`，区别于 PE-29k 的 IEv6 prompt |
| Caption 推理 | 首轮 greedy、seed 42、max model length 32768、max new tokens 27000、trained MTP；截断/JSON/schema 错误明确排除 |
| DiT | `edit2k-r21-sft-ema20k-15n-1009-0652`，step=4000，原生 DCP |
| 权重 | **仅 `state_dict.ema_denoiser.*`，463 键，严格加载；不加载 optimizer** |
| 生图 | trainorder-v2，EMA，CFG=2.5，50 sampling steps，native 2K，max pixels 4194304，timeshift=4.6，seed 42 |
| H100 encoder | Float8CurrentScaling；不是 B300 MXFP8 的逐位复现 |
| GSB | gemini-3.8-flash、LOW、temperature 0、AB/BA 双向；一致性与视觉质量；原始指令评分 |
| UTP | 4 indexed CPU workers，每 worker 256 并发；总并发上限 1024；每 worker 80 CPU / 180Gi |

配置中的 checkpoint、模型导出、输入、prompt 和源码包均以 SHA256 固定，不能用 initializer 替代 trained checkpoint。PE 步数与 DiT 步数分别记录，`checkpoint-3000` 不表示 DiT step 3000。

```text
s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/runs/c29k-step0-b300128-w500/20261009-39c59fafbc35/mcore/checkpoint-3000/
s3://core-cn-training.canva.com/pixelcraft/usr/tayo/runs/edit2k-r21-sft-ema20k-15n-1009-0652/checkpoints/step=4000.ckpt/
```

Qwen 改写不是本次重新生成：1,464 条冻结改写的 SHA 为 `528ee56b1090a2c4841a661057cf863778618e18f10a2a7b6a20393828dd56c2`。如需新 Qwen 预测，使用根目录 [handover.md 第 5 节](../../handover.md#5-stage-a可选-qwen-pe-改写) 的独立入口，并重新建立输入锁；本目录不接受静默替换冻结输入。

## 2. 已验证结果与消费入口

| Profile | Caption 全量 / valid | DiT 图片 | GSB 判断 |
|---|---:|---:|---:|
| PE-3k + Qwen-PE | 1,464 / 1,390 | 1,258 | 2,516 |
| PE-4k + Qwen-PE | 1,464 / 1,384 | 1,256 | 2,512 |

PE-3k 的 206 个排除项：74 个 caption 错误、88 个不支持的多 reference box、44 个 native 空间/token 限制。`valid.jsonl` 仅表示 caption JSON/schema 合格，不等于 native admission 通过。

```bash
export PE3_CAP=s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/pe39c59-c3000-qwenpe-caption-euc1-h100128-20261009
export PE3_IMG=s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/pe39c59-c3000-qwenpe-dit4k-s4000-cfg25-euc1-h100128-20261009
```

| 内容 | 路径 |
|---|---|
| 完整 JSON caption | `${PE3_CAP}/caption/predictions.jsonl` |
| JSON/schema 合格 caption | `${PE3_CAP}/caption/valid.jsonl` |
| 实际生图使用的 caption | `${PE3_IMG}/caption/admitted.jsonl` |
| 所有排除原因 | `${PE3_IMG}/caption/excluded.jsonl` |
| 图片消费入口 | `${PE3_IMG}/editor/score-index.jsonl` |
| 图片 hash / URI 清单 | `${PE3_IMG}/editor/images.manifest.jsonl` |
| 图片文件 | `${PE3_IMG}/nodes/00..15/editor/images/`，按 index 中 URI 读取 |
| GSB | `${PE3_IMG}/gsb-vs-qwen21/{scores.jsonl,summary.json,metrics.csv,VERIFIED.json}` |
| 顺序检查 | `${PE3_IMG}/execution/encoder-order-audit.json` |

PE-4k 的对应前缀见 [profiles/pe4k.json](profiles/pe4k.json)。完成证据见 [evidence/](evidence/)。历史结果不可覆盖；恢复脚本拒绝向这些已有结果前缀写入。

## 3. 环境与恢复

此执行拓扑固定为 **16 个已授权 H100 holder**，每节点 8 GPU；node 2 / GPU 4 因现场健康问题排除，实际使用 127 张。不要把示例节点序号映射到别人的 Pod。新拓扑、题库或硬件需要重新验证，不能直接修改数字后视为同一实验。

Pod 中需有 Python 3.12、boto3、有效的 `core-cn-s3-readwrite` profile、已验证 caption 环境和 H100 native runtime。caption 环境为 torch 2.11.0 / vLLM 0.26.0 / transformers 5.8.0；native 环境为 torch `2.13.0a0+8145d630e8.nv26.06`。环境恢复步骤沿用根目录 [handover.md 第 4 节](../../handover.md#4-h100-环境权限和恢复)。native runtime 固定布局 `/mnt/ephemeral/pcb-v2-9c4d-editor-runtime`，使用 `/opt/runtime/bin/python -S`。本入口复用已恢复环境，不在运行中升级依赖。

以下变量在 16 个 Pod 中使用相同值。将 `YOUR_NAME`、`YOUR_UNIQUE_RUN` 与 `RUNTIME_FROM` 替换成自己的已授权资源；workspace 必须是新的节点本地目录。

```bash
export PE_TOOLS="$PWD/pipelines/pe_qwenpe_native2k"
export PE_PROFILE=pe3k
export PE_WORK=/mnt/ephemeral/YOUR_UNIQUE_RUN
export PE_OUT=s3://core-cn-oss.canva.com/usr/YOUR_NAME/pixelcraftbench-v2/evaluations/YOUR_UNIQUE_RUN
export RUNTIME_FROM=/mnt/ephemeral/YOUR_VERIFIED_RUNTIME_CAMPAIGN

python3 "$PE_TOOLS/materialize.py" --profile "$PE_PROFILE" --stage caption \
  --workspace "$PE_WORK" --output-s3 "$PE_OUT" --runtime-from "$RUNTIME_FROM"
```

恢复脚本下载 profile 中的原始部署 tar，核对整体 SHA256 和 `SOURCE_MANIFEST.json`，然后仅配置新输出位置及本地依赖。模型来源继续使用已校验的 S3 URI。`CONFIGURED.json` 记录配置后的 campaign SHA；配置后禁止在同一工作目录切换 profile。`recorded/` 提供实际 worker、native runner、Luna prompt 和 PE export 代码，完整依赖由固定 tar 恢复。

## 4. Qwen-PE → PE JSON caption

先在 node 0 执行四样本 canary。检查通过后自动写入本次输出前缀的 `CANARY_PASS.json`。其他节点的 full run 等待同一 campaign SHA 的 canary 标记。

```bash
# 仅 node 0。
python3 "$PE_TOOLS/run_node.py" run --root "$PE_WORK/caption" --node 0 --canary

# 每个节点分别运行一次；NODE_INDEX 为其固定序号 0..15。
python3 "$PE_TOOLS/run_node.py" run --root "$PE_WORK/caption" --node "$NODE_INDEX"

# 仅 node 0；等待 16 个节点完成并进行 S3 完整回读校验。
python3 "$PE_TOOLS/run_node.py" finalize --root "$PE_WORK/caption"
```

执行保留 `original_instruction`；PE 接收 Qwen `rewritten_prompt` 原文。缺失、重复、空改写和身份不匹配会失败。无补采样，也不静默退回原指令。导出模型保留训练后的 MTP 及 FP32 `A_log` / `dt_bias`；导出源码仅用于审计，常规推理直接下载 profile 固定的 trained-HF 权重。

## 5. Caption → native DiT 4k 生图

每个节点仍须保留刚完成的本地 caption 工作目录：native 阶段逐节点检查 caption 文件 SHA、S3 checksum、PE checkpoint 和输入身份，再复制同一份 caption。

```bash
# 在 16 个节点恢复 editor。这里的 dit4000 与 PE_PROFILE 的 3k / 4k 独立。
python3 "$PE_TOOLS/materialize.py" --profile "$PE_PROFILE" --stage dit4000 \
  --workspace "$PE_WORK" --output-s3 "$PE_OUT" --runtime-from "$RUNTIME_FROM"

# 仅 node 0：EMA 严格恢复、数值有限性、图像回执与提示词顺序 canary。
python3 "$PE_TOOLS/run_node.py" run --root "$PE_WORK/dit4000" --node 0 --canary

# 每个节点一次；脚本沿用 GPU 排他锁和占用检查。
python3 "$PE_TOOLS/run_node.py" run --root "$PE_WORK/dit4000" --node "$NODE_INDEX"

# 仅 node 0：汇总、核对 127 份 EMA restore 证据、图片校验和、提示词顺序。
python3 "$PE_TOOLS/run_node.py" finalize --root "$PE_WORK/dit4000"
```

**Text encoder 顺序不是字母序排序。** Adapter 恢复训练 schema 的 insertion order，保留 layout/action/palette 等列表顺序。native 根字段为 `normalized_edit_instruction → high_level_description → style_description → compositional_deconstruction`。不能对送入 encoder 的 caption 使用 `sort_keys=True`。每张图片都校验实际 positive/weak prompt SHA 与 native audit、生成 receipt 一致。GSB/provenance 自身用 `sort_keys=True` 做确定性 hash，不会改变 encoder caption。

## 6. UTP 4 workers × 256 GSB

以下操作在能够访问结果目录、S3 和 Arnold 的环境执行。若从 Pod 转到 devbox，复制两个阶段的 `final/` 与 `campaign.json`，保留 `results -> final` 的链接；无需复制图片或 GPU 模型。

```bash
export GSB_JOB=YOUR_UNIQUE_GSB_JOB
python3 "$PE_TOOLS/prepare_gsb.py" --profile "$PE_PROFILE" --workspace "$PE_WORK" \
  --job-name "$GSB_JOB"

arnold submit -f "$PE_WORK/utp/jobset.yaml" --application-id core-cn \
  --project-id pixelcraftbench --cluster p-iad-eagle --namespace b-core-cn-prod \
  --job-name "$GSB_JOB" --dry-run

arnold submit -f "$PE_WORK/utp/jobset.yaml" --application-id core-cn \
  --project-id pixelcraftbench --cluster p-iad-eagle --namespace b-core-cn-prod \
  --job-name "$GSB_JOB" --no-wait
```

`prepare_gsb.py` 本身不提交任务。它校验生成完成证明和顺序审计，恢复冻结 Qwen 基线与 rubric，构建四个互斥且覆盖全部样本的分片，并上传 hash 固定的评分包。新 JobSet 必须使用唯一名称。

每个样本的 AB、BA 归同一个 worker。两个方向一致时才判 G/B，否则判 S。评分使用原始 instruction/source，不使用改写指令、PE JSON 或 GT 图。图片统一 RGB、面积上限 1048576、Lanczos、WebP quality 90。评分核心函数 AST 与原始 worker 一致。

JobSet 使用四个持续工作的 Indexed worker，每个 80 CPU / 180Gi，合计 320 CPU / 720Gi、0 GPU；on-demand 24/32/48xlarge，required 跨 namespace 单 trainer/node，Core CN queue、model-trainer SA、独立 EFS 目录及 Vertex WIF。不得降低资源请求绕过队列或移除 anti-affinity。

持久化进度在 `${PE_OUT}/dit4000/gsb-vs-qwen21/judgments/progress/worker-0..3.json`。最终读取 `gsb-vs-qwen21/VERIFIED.json`，仅 `status=PASS` 且 4 个 worker 完整结束后报告正式结果。失败时保留原协议和结果，复用已持久化判断；不要重新冻结协议覆盖同一前缀。

净胜率为 `(G−B)/N`；共有样本比较先按 `sample_id` 取交集，再检查 source、control、original instruction 和 input index 一致。2,511 次完成调用不能称作 2,511 个 GSB 样本：若有一次 AB/BA 未完成，只报告完整配对的临时结果。

## 7. PE-4k caption 切换 DiT 6k

仅 `pe4k` profile 提供本轮 DiT 6000 bundle。先完成同一 workspace 中的 caption 与 DiT4000，再使用 `--stage dit6000` 重复第 5 节。保留本地 DiT4000 bundles：只有完整 resume contract、caption、native manifest、audit 和 encoding identity 全部一致才复用编码。所有图片重新使用 step=6000 的 463 个 EMA 权重生成；CFG、50 sampling steps、timeshift、seed 和 canvas 保持不变。

当前 GSB 准备入口明确限定 **DiT4000**；DiT6000 需要独立配置、结果前缀及评分协议，不能冒用 DiT4000 的指标。

## 8. 验证边界

PE-3k/PE-4k 的 caption、生图和 UTP GSB 历史完整执行已有 S3 校验依据。新恢复入口使用同一模型/推理/judge 源码，另外执行本地 bundle 恢复、样本/模型锁和四分片覆盖检查；新 workspace 的 GPU 运行仍必须通过自身 canary。Arnold 3.10.3 的 PE-3k JobSet dry-run 也已通过（未提交新作业）。详见 `VALIDATION.json`。

源码交接校验：`python3 pipelines/pe_qwenpe_native2k/verify_release.py`。该命令无需凭证，不执行 GPU 推理或提交作业。

不打包个人 kubeconfig、Teleport/Roo 会话、AWS/Google 凭证、模型权重或生成图片。目录里的 S3 URI 需要使用者自己的有效权限。
