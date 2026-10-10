# PE-3k + Qwen-PE：greedy → 重复规则重试 → DiT 6k → UTP 4×256 GSB

本目录对应 2026-10-10 完成的独立实验：复用 1,464 条冻结 Qwen-PE 改写，重新生成全部 PE JSON caption；先 greedy，仅对重复规则命中的样本采样重试一次，再重新编码并使用 DiT 6k 生图。**1,275 张图片、2,550 次 AB/BA 判断已完成，生成、text encoder 顺序、GSB 全量审计均 PASS。**

这是新增的 caption 策略实验。之前[复用旧 caption 切换 DiT 6k 的实验](../PE3K.md#复用同一-caption-切换-dit-6k)保持独立，其中 1,257 个评分样本和用户要求的单样本排除不适用于本次。

## 配置与重复规则

模型与源码锁见 [profile.json](profile.json)，历史实际配置见 [recorded/campaign.json](recorded/campaign.json)。

| 阶段 | 配置 |
|---|---|
| PE | W&B `39c59fafbc35` / checkpoint-3000，trained-MTP HF export |
| Qwen-PE | 冻结改写 SHA256 `528ee56b1090a2c4841a661057cf863778618e18f10a2a7b6a20393828dd56c2`；本次没有重新调用 Qwen 改写 |
| PE prompt | `luna_source_target_size_ordered_sparse_reference_edits_v1`；正确 non-thinking 前缀 |
| PE decoding | seed=42、max_model_len=32768、max_new_tokens=27000、max_pixels=2359296 |
| DiT | `edit2k-r21-sft-ema20k-15n-1009-0652/checkpoints/step=6000.ckpt/` 原生 DCP |
| 权重 | `state_dict.ema_denoiser.*`，463 键，strict restore |
| 生图 | EMA、CFG=2.5、50 steps、native 2K、max_pixels=4194304、timeshift=4.6、seed=42 |
| Encoder | trainorder-v2；H100 Float8CurrentScaling |
| GSB | Gemini 3.8 Flash / LOW / temperature=0；AB/BA；Qwen-Image 2.1 CFG1 对照 |
| UTP | 4 indexed CPU workers，每个 256 并发上限；每个 80 CPU / 180Gi |

| 采样参数 | 首轮 greedy | 规则命中后唯一一次重试 |
|---|---:|---:|
| temperature | 0 | 0.7 |
| top_p | 1 | 0.8 |
| top_k | −1 | 20 |
| min_p | 0 | 0 |
| presence_penalty | 0 | 1.5 |
| repetition_penalty | 1 | 1 |

两种模式均 frequency_penalty=0、ignore_eos=false。首轮达到以下任一条件才重试：

1. 连续重复同一段 region 序列至少 3 轮，累计至少 6 个 region；比较完整 region 内容，仅忽略该 region 自身的 `label`，保留 bbox、parent、文本与描述。检测周期为 1–32 个 region。
2. 同一 JSON 字符串内，词串连续重复至少 8 次，累计至少 128 个词法单元；周期为 1–256 个词法单元。词法单元按 `\w+(?:[-']\w+)*|[^\w\s]` 划分，包含标点，不等同于模型 token。可扫描截断输出中的未闭合字符串。

检测在每次生成结束后进行，不会实时打断解码。未命中规则的截断或 JSON/schema 错误不自动重试。重试仍命中、截断或 schema 不合格的结果保留原始记录并排除生图；另有原生 adapter 和空间/token admission 检查。

**阈值仍需人工标注样本校准，命中率不能当作准确率。有限输出中检测到重复模式，也不等于证明模型会无限循环。** 检测仅生成审计信息，不重排或去重实际 caption。代码：[规则](recorded/package/workflow/code/repetition_rules.py)、[greedy/retry 编排](recorded/adaptive_caption.py)、[采样入口](recorded/package/workflow/code/infer_caption.py)。

## 已完成结果

| 项目 | 数量 |
|---|---:|
| 新生成 greedy caption | 1,464 |
| 规则命中并重试 | 37（region 29，字符串 8） |
| 重试后不再命中规则 | 32 |
| 重试后规则通过且 JSON/schema 合格 | 27 |
| 重试样本最终成功生图 | 19 |
| 重试后仍命中，已排除 | 5 |
| 最终 JSON / schema 合格 | 1,436 / 1,410 |
| 最终生图 / 排除 | 1,275 / 189 |

189 个排除项：不支持的多 reference box 93，native 空间/token 上限 40，重试后重复 5，caption JSON/schema 错误 25，caption 截断 26。这是最终互斥归类，不能将各阶段独立错误计数直接相加。

本次使用全部 128 张已授权 H100。node 2 / GPU 4 后续发生热降频，21 张剩余图片迁移到该节点其余 7 张 GPU；60 张已完成图片的 SHA 保持不变。记录见 [THERMAL_RECOVERY_VERIFIED.json](evidence/THERMAL_RECOVERY_VERIFIED.json)。新运行要求健康节点，不自动执行本次现场迁移。

GSB 全量 **N=1,275**，相对冻结 Qwen-Image 2.1：

| 指标 | Good | Similar | Bad | 净胜率 |
|---|---:|---:|---:|---:|
| 指令一致性 | 384 | 385 | 506 | −9.57% |
| 画面质量 | 303 | 406 | 566 | −20.63% |

与 PE-3k + Qwen-PE + DiT 4k / CFG2.5 的 **1,256 个共有样本**比较：

| 指标 | 旧 caption + DiT4k G/S/B | 新 caption + DiT6k G/S/B | 4k 净胜率 | 6k 净胜率 | 差值 |
|---|---|---|---:|---:|---:|
| 指令一致性 | 376/379/501 | 379/381/496 | −9.95% | −9.32% | +0.64 pp |
| 画面质量 | 295/399/562 | 301/401/554 | −21.26% | −20.14% | +1.11 pp |

净胜率 `(G−B)/N`，差值使用未四舍五入数值计算。两组分别对完全相同的 Qwen 对照图评分，已核对原图、指令、judge 协议和样本身份；不是两组图片的直接对战。caption 同时重新生成并加入规则重试，因此差值不能单独归因于 DiT checkpoint。

证据：[生成汇总](evidence/generation-summary.json)、[重复审计](evidence/repetition-summary.json)、[顺序审计](evidence/encoder-order-summary.json)、[GSB 全量汇总](evidence/gsb-summary.json)、[GSB 验证](evidence/gsb-VERIFIED.json)、[共有样本比较及 ID](evidence/dit4k-comparison.json)。

## 恢复与运行

需要自己的 S3 权限、16 个已授权且空闲的 8×H100 节点，以及[基础流程](../README.md#3-环境与恢复)中的已验证 caption/native 环境。底层 native runtime 仍使用固定路径 `/mnt/ephemeral/pcb-v2-9c4d-editor-runtime`。每个节点须具备同一个经过 SHA256 核验的 DiT6000 cache；各节点传入相同路径与配置，以保持 campaign SHA 一致。

恢复入口下载 [profile.json](profile.json) 中固定的 19,573,708 字节生成源码包，并验证整体 SHA256 和包内 1,047 个文件的原始 `SOURCE_MANIFEST.json`。模型、输入和推理源码保持冻结；配置新的输出前缀及可选环境/cache 路径。历史 bundle 的三个遗留描述字段在新 workspace 中修正，原始源码包和[现场说明](evidence/execution-manifest-notes.json)保留供审计。

```bash
# 每个节点各执行一次，使用相同值；替换 YOUR_*。
export ADAPTIVE_TOOLS="$PWD/pipelines/pe_qwenpe_native2k/adaptive_pe3k"
export ADAPTIVE_WORK=/mnt/ephemeral/YOUR_UNIQUE_ADAPTIVE_RUN
export ADAPTIVE_OUT=s3://core-cn-oss.canva.com/usr/YOUR_NAME/pixelcraftbench-v2/evaluations/YOUR_UNIQUE_ADAPTIVE_RUN
export RUNTIME_FROM=/mnt/ephemeral/YOUR_VERIFIED_RUNTIME_CAMPAIGN
export DIT6K_CACHE=/mnt/ephemeral/YOUR_VERIFIED_DIT6K_CACHE
python3 "$ADAPTIVE_TOOLS/restore.py" --workspace "$ADAPTIVE_WORK" \
  --output-s3 "$ADAPTIVE_OUT" --runtime-from "$RUNTIME_FROM" \
  --dit-cache-from "$DIT6K_CACHE"

# 仅 node 0：4 个 caption canary，覆盖两种规则及无需重试的样本；然后生图/顺序校验。
python3 "$ADAPTIVE_TOOLS/run.py" run --root "$ADAPTIVE_WORK/run" --node 0 --canary

# 16 个节点分别运行；NODE_INDEX 是该节点固定序号 0..15。
# 完整运行必须通过本次 campaign SHA 对应的 canary。
python3 "$ADAPTIVE_TOOLS/run.py" run --root "$ADAPTIVE_WORK/run" --node "$NODE_INDEX"

# 仅 node 0：等全部节点完成后，汇总并验证 128 份 EMA restore 证据与 S3 图像回执。
python3 "$ADAPTIVE_TOOLS/run.py" finalize --root "$ADAPTIVE_WORK/run"
```

`run.py` 执行冻结 worker，检查每个最终 caption 的规则路由、采样和响应 hash，以及实际 encoder prompt hash 与图片回执的绑定。`trainorder-v2` 恢复训练 schema 字段顺序并保留 region 原始顺序，不进行字母序排序。每次运行独立生成 PE caption、conditioning 和图片；只复用冻结 Qwen 改写及已校验模型资产。

以上恢复命令保留固定 16 节点拓扑。改变拓扑、模型、prompt 或输入需要另行验证。源码恢复和 GSB 构建可在 CPU 上验证；新 workspace 的 GPU 运行必须执行自身 canary，不能将历史结果当作新运行成功证明。

## UTP 4×256 GSB

生成完成且 S3、encoder 顺序验证通过后，在具有 AWS 权限的执行环境准备独立评分包，再通过 Arnold 提交。无需 GPU；worker 使用 AWS workload identity 经 Vertex WIF 调用 Gemini。

```bash
export GSB_JOB=YOUR_UNIQUE_ADAPTIVE_GSB_JOB
python3 "$ADAPTIVE_TOOLS/prepare_gsb.py" --workspace "$ADAPTIVE_WORK" --job-name "$GSB_JOB"
arnold submit -f "$ADAPTIVE_WORK/utp/jobset.yaml" --application-id core-cn \
  --project-id pixelcraftbench --cluster p-iad-eagle --namespace b-core-cn-prod \
  --job-name "$GSB_JOB" --dry-run
arnold submit -f "$ADAPTIVE_WORK/utp/jobset.yaml" --application-id core-cn \
  --project-id pixelcraftbench --cluster p-iad-eagle --namespace b-core-cn-prod \
  --job-name "$GSB_JOB" --no-wait
```

本次历史任务为 `grantzhang-pe3k-retry-dit6k-gsb4x256-1010`，4 个分片分别有 319/319/319/318 对样本。每个样本必须完成 AB 和 BA 才计分；两个方向不一致计 Similar。保持原 rubric、图像归一化和 judge 实现，`recorded/gsb/code/worker.py` SHA256 为 `cd6c24a4ca68de0911d7be79bd2449b60a889d159836b8ed1a576f96c866c244`。

`prepare_gsb.py --no-upload` 可仅构建和检查，但不能提交尚未上传的 bundle。四个 workers 完成后，rank 0 自动合并并审计所有原始判断，发布 `scores.jsonl`、`summary.json`、`metrics.csv`、`VERIFIED.json`；只在全量验证成功后写完成状态。

## 历史 S3 与本地验证

```text
s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/pe39c59-c3000-qwenpe-greedy-loop-retry-dit6k-cfg25-euc1-h100128-20261010/
```

相对于此前缀：

| 内容 | 路径 |
|---|---|
| 首轮、重试、最终 caption | `caption/greedy-predictions.jsonl`、`caption/retry-predictions.jsonl`、`caption/predictions.jsonl` |
| 生图准入 / 排除 | `caption/admitted.jsonl`、`caption/excluded.jsonl` |
| 图像索引 | `editor/score-index.jsonl`、`editor/images.manifest.jsonl` |
| 重复检测审计 | `repetition-audit/summary.json` 及各轮 `*.audit.jsonl` |
| 实际 prompt 顺序核验 | `execution/encoder-order-audit.json` |
| GSB | `gsb-vs-qwen21/{scores.jsonl,summary.json,metrics.csv,VERIFIED.json,PUBLICATION_VERIFIED.json}` |
| 完整源码包与 SHA256 | [profile.json](profile.json) 的 `bundles.generation` / `bundles.gsb` |

`recorded/` 是实际执行源码快照，用于审阅；运行须通过上面的新 workspace 入口，避免直接执行快照中的历史输出前缀。仓库不包含个人会话、凭证、模型权重或生成图片。

```bash
python3 pipelines/pe_qwenpe_native2k/verify_release.py
```

完整 GPU 推理和评分证据来自已完成的历史实验。新增恢复包装入口另外完成本地源码包恢复、配置隔离、评分包重建、四分片完整覆盖和 Arnold dry-run 验证，详见 [VALIDATION.json](VALIDATION.json)。本次发布没有重新提交 GPU 或评分任务。
