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
