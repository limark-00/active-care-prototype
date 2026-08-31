# 训练说明

代码包含三个可比较模型：

1. `train_baseline.py`：字符TF-IDF + 四个SGD逻辑分类器。
2. `train_event_transformer.py`：冻结`hfl/chinese-macbert-base`，将上下文和每条事件分别编码，再训练两层Event Transformer及四个分类头。
3. `train_partial_macbert.py`：解冻MacBERT最后两层，与Event Transformer和四个分类头联合训练；默认使用不暴露L/I代码、按事件定义隔离的V3数据。

MacBERT约需下载412 MB权重。脚本只使用`input_text`；不会把`reason_codes`或目标标签作为输入。

## 真正的部分微调（下一阶段）

先生成V3，再运行部分微调：

```bash
vision/.venv/bin/python -m decision_model.generate_dataset_v3
bash decision_model/run_partial_finetuning.command
```

默认配置适合16 GB Apple Silicon：批大小4、梯度累积4次（有效批大小16）、解冻最后2层、最多6轮、连续2轮没有提升就提前停止。正式运行预计明显慢于冻结编码实验，具体时间以`train.jsonl`和`run.json`记录为准，不承诺固定分钟数。

如出现MPS内存不足，先把批大小改为2并把梯度累积改为8：

```bash
vision/.venv/bin/python -m decision_model.training.train_partial_macbert \
  --batch-size 2 --gradient-accumulation 8
```

部分微调输出目录以`partial-macbert-`开头。`best_delta.pt`只保存解冻的MacBERT层、Event Transformer和分类头；加载时仍需要缓存中的基础MacBERT。

## V3.1安全边界实验

V3第一轮在严格测试集上出现I3召回率为0的问题。V3.1保留该结果作为基线，增加以下实验变量：

- 每个场景和干预等级都有互斥的训练、验证、测试事件定义。
- 增加`derived_risk_level`辅助分类头。
- 对风险和干预等级增加有序距离损失，并对低估等级增加额外代价。
- 最佳检查点按安全加权分数选择：干预65%、风险15%、警示10%、人工核查5%、拒绝自动判断5%。
- 指标新增平均等级误差、相邻等级准确率、欠干预率、过度干预率和严重欠干预率。

运行：

```bash
bash decision_model/run_partial_finetuning_v31.command
```

输出目录以`partial-macbert-v31-`开头，不会覆盖V3模型。

## 一键运行

从项目根目录执行：

```bash
bash decision_model/run_training.command
```

脚本复用`vision/.venv`中的PyTorch，安装scikit-learn、Transformers、joblib和tqdm，然后依次训练两个模型。M2 Pro 16 GB的预估时间：基线约1～5分钟；首次下载和冻结编码约20～90分钟；Event Transformer约10～40分钟。网络或MPS回退到CPU时可能更久。

## 分开运行

```bash
vision/.venv/bin/python -m pip install -r decision_model/training/requirements.txt
vision/.venv/bin/python -m decision_model.training.train_baseline
vision/.venv/bin/python -m decision_model.training.train_event_transformer --device auto
```

不要关闭终端或让电脑进入睡眠。训练不会开启摄像头，也不会上传数据。MacBERT首次从Hugging Face下载并缓存在本机。

## 结果

每次运行创建独立时间戳目录：

```text
decision_model/outputs/
├── baseline-时间/
│   ├── model.joblib
│   ├── run.json
│   └── metrics-*.json
└── event-transformer-时间/
    ├── best_model.pt
    ├── encoder.json
    ├── train.jsonl
    ├── run.json
    └── metrics-*.json
```

训练日志每个epoch写入`train.jsonl`。需要分析时保留整个运行目录。模型选择只使用validation，`test`、`ood_test`和`natural_test`只在训练结束后报告。
