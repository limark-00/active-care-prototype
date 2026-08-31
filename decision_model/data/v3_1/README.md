# 合成文字决策数据集 V3.1

V3.1针对V3暴露出的I3泛化失败进行修订。数据仍为合成策略实验，不代表真实照护、医学或紧急处置标准。

## 相对V3的变化

- 新增42个独立事件定义，集中补充I2、I3和I4边界。
- 共222个事件定义。
- 每个“场景 × 干预等级”在训练、验证和测试中都有独立事件。
- 三个主分区的事件ID完全互斥。
- `input_text`仍不显示L/I答案代码。
- 保留`derived_risk_level`，供新的风险辅助预测头监督使用。

## 数据规模

| 文件 | 样本数 |
| --- | ---: |
| `train.jsonl` | 4,800 |
| `validation.jsonl` | 1,200 |
| `test.jsonl` | 1,200 |
| `ood_test.jsonl` | 600 |
| `natural_test.jsonl` | 600 |

重新生成：

```bash
vision/.venv/bin/python -m decision_model.generate_dataset_v31
```

`natural_test`复用测试分区的事件定义并改变语言风格，用于测量表述扰动；它不是独立事件测试。`test`负责未见事件定义评估，`ood_test`使用另外30个保留事件。
