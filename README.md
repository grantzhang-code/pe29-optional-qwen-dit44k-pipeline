# Multi-reference PE · checkpoint-8000

本分支提供训练 run `e6d88249d08d` 的 **checkpoint-8000 多图 PE 推理 pipeline**：有序 reference images + 原始 instruction + 目标画布 → JSON caption。

**入口：[multi_reference/README.md](multi_reference/README.md)**

- 支持 1–12 张 reference，冻结训练 prompt、trained MTP 和 S3 checkpoint 身份。
- 支持 419 条冻结 MDS 评测或自定义 S3 多图 JSONL。
- 提供模型恢复、输入准备、单卡/多卡推理、schema 重试、合并和 S3 回读验证。
- 保留历史原始结果、targeted recovery 和结构修复的独立说明。

以下为同仓库保留的单图 PE29 pipeline 说明。

---

# PE-29k · Optional Qwen-PE · Fixed DiT44k

完整的 image-edit 推理交接：用户 instruction → 可选 Qwen-PE 改写 → PE-29k JSON caption → 修复版 DiT44k 编辑生图。

本仓库对应 **2026-10-09 v2**，保留 S3 发布包的原始内容与 SHA256 校验文件。

## 两种 instruction 模式

| 模式 | PE-29k 接收的 instruction |
|---|---|
| 默认 / `--no-use-qwen-pe` | 用户原始 instruction 原文 |
| `--use-qwen-pe` | Qwen-PE 的 `rewritten_prompt` 原文替换 instruction |

两种模式都保留 `original_instruction`，并保持样本身份、source 图和目标画布不变。Qwen 开启时，缺失、重复、空输出、截断或身份不匹配会报错，不会自动退回原指令。

## 使用入口

完整的环境恢复、Qwen 重跑、caption、native admission、DiT 生图和发布命令见 **[handover.md](handover.md)**。

通过 Git clone 使用源码时，先进入本仓库目录：

```bash
export PE29_ROOT="$PWD/package"
python3 "$PE29_ROOT/verify_package.py"
```

模型来源统一配置在 [checkpoint-sources.json](package/workflow/checkpoint-sources.json)，所有来源必须是 S3 URI。本仓库不包含模型权重；运行需要访问文档中的 S3 资产及相应 GPU/Python 环境。本地模型目录仅用于下载缓存。

将下例中的 `YOUR_NAME` 和 `YOUR_UNIQUE_PE29_RUN` 替换成自己的 S3 结果前缀，然后选择一种模式：

```bash
export PE29_NEW_S3=s3://core-cn-oss.canva.com/usr/YOUR_NAME/pixelcraftbench-v2/evaluations/YOUR_UNIQUE_PE29_RUN

# 默认模式：原始 instruction → PE-29k。
python3 "$PE29_ROOT/configure_pipeline.py" \
  --no-use-qwen-pe --node-count 1 --output-s3 "$PE29_NEW_S3"
```

启用 Qwen 时，用下列命令替代上面的配置命令：

```bash
# 使用随包冻结、经过身份校验的 Qwen-PE 输出。
python3 "$PE29_ROOT/configure_pipeline.py" \
  --use-qwen-pe --node-count 1 --output-s3 "$PE29_NEW_S3"
```

重新生成 Qwen 改写或使用新预测快照，见交接文档第 5 节。完成配置后，按第 4、6–9 节恢复环境并执行推理。运行开始后，更换模式需使用新目录。

## 文件

| 路径 | 内容 |
|---|---|
| [handover.md](handover.md) | 完整交接文档 |
| [package/](package/) | 解包后的完整源码、输入快照及历史结果证据 |
| [workflow-sources.tar.gz](workflow-sources.tar.gz) | 与 S3 发布一致的完整源码包 |
| [package.manifest.json](package.manifest.json) | 发布文件 bytes / SHA256 |
| [LOCAL_VERIFIED.json](LOCAL_VERIFIED.json) | 本地回归与静态验证结果 |
| [S3_SOURCES_VERIFIED.json](S3_SOURCES_VERIFIED.json) | S3 模型前缀及清单验证结果 |
| [VERIFIED.json](VERIFIED.json) | S3 上传回读验证回执 |

## 验证范围

7 项 CPU 回归测试通过，两种模式各覆盖完整 1,464 条输入，并调用真实 PE normalizer。370 个包内文件、320 个 Python 文件及 16 个 Bash 文档代码块完成校验；6 个 DiT 修复核心文件与 v1 一致。

恢复 caption 环境后可运行：

```bash
cd "$PE29_ROOT"
"$PE29_ROOT/assets/env/caption/bin/python" -m unittest discover -s tests -v
```

本次没有重新运行 GPU 推理或完整模型/环境下载。文档中的 1,427 张图片及 GSB 是历史 Qwen 开启实验结果；新运行仍需执行 canary。

## S3 发布位置

```text
s3://core-cn-oss.canva.com/usr/grantzhang/pixelcraftbench-v2/handovers/pe29-optional-qwen-dit44k-fixed-20261009-v2/
```
