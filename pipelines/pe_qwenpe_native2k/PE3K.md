# PE-3k + Qwen-PE 完整流程

训练任务：`39c59fafbc35` / **checkpoint-3000**。使用冻结 Qwen-PE 改写，重新推理 JSON caption，再以 **DiT step=4000 EMA / CFG=2.5 / 50 steps / native 2K / timeshift=4.6** 编辑生图，最后通过 **UTP 4 workers × 256 并发**做 AB/BA GSB。

- [逐步恢复、canary、生图、评分命令](README.md)
- [PE-3k 模型、prompt、输入与源码包锁](profiles/pe3k.json)
- [实际 caption 脚本和配置](recorded/pe3k/caption/)
- [实际 native 编辑脚本和配置](recorded/pe3k/dit4000/)
- [GSB 准备入口](prepare_gsb.py)、[UTP 调度代码](gsb/)
- [已完成结果与校验证据](evidence/pe3k/)

运行说明中的 `PE_PROFILE=pe3k` 已是本配置。16 个已有 H100 holder / 127 个健康 GPU 是该次验证拓扑；新 workspace 必须使用自己的结果前缀并通过 canary。

已完成：1,464 条 caption，1,390 条 JSON/schema 合格；1,258 张图，2,516 次 GSB 判断，最终验证 PASS。

```text
# JSON caption
s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/pe39c59-c3000-qwenpe-caption-euc1-h100128-20261009/caption/predictions.jsonl

# 生图索引
s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/pe39c59-c3000-qwenpe-dit4k-s4000-cfg25-euc1-h100128-20261009/editor/score-index.jsonl

# GSB
s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/pe39c59-c3000-qwenpe-dit4k-s4000-cfg25-euc1-h100128-20261009/gsb-vs-qwen21/
```

与 PE-2k + Qwen-PE 的 **1,195 个共有样本**比较（均为 DiT4k / CFG2.5，分别对同一 Qwen 基线）：

| 维度 | PE-2k G/S/B | PE-3k G/S/B | PE-2k 净胜率 | PE-3k 净胜率 | 3k−2k |
|---|---|---|---:|---:|---:|
| 一致性 | 347/344/504 | 359/360/476 | −13.14% | −9.79% | +3.35 pp |
| 视觉质量 | 281/391/523 | 275/387/533 | −20.25% | −21.59% | −1.34 pp |

净胜率 `(G−B)/1195`。这不是 PE-3k 对 PE-2k 的直接对战。机器可读结果见 [gsb-common-cohorts.json](evidence/pe3k/gsb-common-cohorts.json)。

## 复用同一 caption 切换 DiT 6k

[运行说明第 7 节](README.md#7-pe-3k--pe-4k-caption-切换-dit-6k) 提供 `pe3k / dit6000` 的恢复与运行命令。复用 PE-3k caption 和经过完整身份校验的 DiT4k conditioning bundles，改用 DiT step=6000 的 463 个 EMA 权重重新生成全部图片；CFG=2.5、50 steps、native 2K、timeshift=4.6、seed=42 和 trainorder-v2 保持不变。

源码见 [recorded/pe3k/dit6000/](recorded/pe3k/dit6000/)，完整部署包及 SHA256 见 [profiles/pe3k.json](profiles/pe3k.json)。本页上方的 GSB 指标对应 DiT4k。

DiT6000 生图已完成：**1,258 张，206 条排除记录**，S3 全量校验与 text encoder 顺序审计均 PASS。与 DiT4000 的 caption、样本集合及 conditioning 一致；1,258 张输出均重新生成且 hash 改变。证据见 [生成汇总](evidence/pe3k/dit6000-generation-summary.json)、[顺序审计](evidence/pe3k/dit6000-encoder-order-summary.json) 和 [同 caption 对照](evidence/pe3k/dit6000-same-caption-dit-comparison.json)。

DiT6000 的 GSB 使用 `prepare_gsb.py --profile pe3k --dit-step 6000`；完整命令见运行说明第 7 节。沿用同一 Qwen-Image-2.1 CFG1 基线和 judge，UTP 4×256；1,258 对样本分为 315/315/314/314，合计 2,516 次 AB/BA 判断。

本次评分按用户要求取消最后一条未完成请求，并排除其缺少一个方向的样本。正式统计 **1,257 个完整 AB/BA 样本、2,514 次判断**，审计 PASS 仅适用于此范围；原始计划为 1,258 个样本。已保存的 2,515 条原始判断全部保留，其中被排除样本的单方向结果不计入指标。原协议未修改，任务已取消；后续仅汇总已保存判断。详见 [排除说明](evidence/pe3k/dit6000-gsb-exclusion-summary.json)、[验证回执](evidence/pe3k/dit6000-GSB_VERIFIED.json) 和 [只汇总已有判断的脚本](recorded/pe3k/dit6000_gsb/finalize_authorized_partial.py)。

与 PE-3k + DiT4000 的相同 **1,257 个样本**比较，均分别对同一 Qwen 基线评分：

| 维度 | DiT4000 G/S/B | DiT6000 G/S/B | DiT4000 净胜率 | DiT6000 净胜率 | 6k−4k |
|---|---|---|---:|---:|---:|
| 一致性 | 377/378/502 | 372/380/505 | −9.94% | −10.58% | −0.64 pp |
| 视觉质量 | 296/399/562 | 310/378/569 | −21.16% | −20.60% | +0.56 pp |

机器可读结果：[DiT6000 GSB 指标](evidence/pe3k/dit6000-gsb-metrics.json)、[共有样本比较](evidence/pe3k/dit6000-gsb-common-cohorts.json)。S3 生图前缀下的 `gsb-vs-qwen21/` 包含 `scores.jsonl`、`summary.json`、`metrics.csv`、`VERIFIED.json` 和明确记录排除样本的 `EXCLUSIONS.json`。

```text
s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/evaluations/pe39c59-c3000-qwenpe-dit6k-s6000-cfg25-euc1-h100128-20261010/editor/score-index.jsonl
```
