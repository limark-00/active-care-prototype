# 合成文字决策数据集 V3

V3 是用于 MacBERT 部分微调的困难版合成数据。它仍然不包含真实患者数据，也不代表医学、照护、消防或紧急处置标准。

相对 V2，V3 有三个关键变化：

1. `input_text` 不再出现 `L0–L4` 或 `I0–I4` 代码，模型必须从事件语义、患者能力、回应和历史干预中学习。
2. 训练、验证和测试按事件定义隔离。同一事件的不同改写不会跨分区，测试集使用的事件定义未出现在训练集。
3. 输入混合正式记录、口语、简写、乱序信息和少量可控错字；这些仍是程序化改写，不能冒充真人语料。

生成后的规模：

| 文件 | 样本数 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 6,000 | 联合训练 |
| `validation.jsonl` | 750 | 模型选择与提前停止 |
| `test.jsonl` | 750 | 未见事件定义测试 |
| `ood_test.jsonl` | 600 | 额外 OOD 事件测试 |
| `natural_test.jsonl` | 约 400 | 口语、简写、错字挑战集 |

重新生成：

```bash
vision/.venv/bin/python -m decision_model.generate_dataset_v3
```

训练脚本只读取 `input_text`。`event_ids`、`event_levels`、`reason_codes` 和目标标签只用于监督、评估和审计，不会送入文字编码器。
