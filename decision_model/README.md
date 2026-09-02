# AI文字决策模型：合成数据与训练

本目录为独立文字决策模型准备可复现的合成数据。输入由家庭场景、多条文字事件、患者能力、患者响应和前次干预组成；输入文字不包含L/I答案代码。V3.1同时学习风险、干预、警示、人工核查和拒绝自动判断五个任务。

全部样本均为虚构数据，不含真实患者信息。标签来自代码中可审计的原型策略，只适合软件流程、模型训练和工业设计演示，不是医学、照护、消防或紧急处置标准。

## 数据版本

| 版本 | 用途 | 规模 |
| --- | --- | ---: |
| [V1](data/v1/dataset_summary.json) | 初始基线；108个事件、三种排列 | 3,600条 |
| [V2](data/v2/README.md) | 大规模基线；180个事件、反事实、冲突和分布外测试 | 32,400条 |
| [V3](data/v3/README.md) | 事件定义互斥切分的困难版部分微调数据 | 8,400条 |
| [V3.1](data/v3_1/dataset_summary.json) | 当前安全边界版；222个事件定义、五任务联合学习 | 8,400条 |

V3.1包含4,800条训练、1,200条验证、1,200条测试，以及各600条未见事件和自然表达测试。训练场景为厨房、卫生间、卧室、客厅、入户门区域和阳台；其他地点属于分布外输入。I0～I4在主要分区中各1,440条。

## 模型训练

训练代码和参数说明见 [training/README.md](training/README.md)。项目包含TF-IDF基线、冻结MacBERT文字编码器后的两层Event Transformer，以及解冻MacBERT最后两层的V3.1联合部分微调。

从项目根目录生成并训练V3.1：

```bash
python3 -m decision_model.generate_dataset_v31
python3 -m unittest decision_model.tests.test_dataset_v31 -v
bash decision_model/run_partial_finetuning_v31.command
```

检查点写入`outputs/partial-macbert-v31-时间/best_delta.pt`。网页服务默认选择最新V3.1检查点，也可用`CARE_DECISION_CHECKPOINT`明确指定。

## 直接安装已训练的V3.1模型

Git仓库保留数据、训练代码和推理结构，但不把二进制训练输出写入普通提交。已训练的V3.1增量权重由GitHub Release分发，并通过SHA256校验。

macOS在项目根目录运行：

```sh
bash 下载模型.command
```

Windows双击`下载模型-Windows.bat`，或在PowerShell运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\下载模型-Windows.ps1
```

下载器同时准备固定版本的`hfl/chinese-macbert-base`。V3.1检查点只保存解冻层、Event Transformer和任务头，缺少基础MacBERT时不能独立推理。完整安装、离线验证和分平台启动说明见项目根目录README。

重新生成V2或V1：

```bash
python3 -m decision_model.generate_dataset_v2
python3 -m unittest decision_model.tests.test_dataset_v2 -v

python3 -m decision_model.generate_dataset
python3 -m unittest decision_model.tests.test_dataset -v
```

生成器使用固定随机种子。同一版本必须产生完全一致的数据；修改事件、策略、表达方式或种子时，需要更新版本并重新检查所有分区。

## 训练时的字段边界

只使用`input_text`作为模型输入。不得输入`reason_codes`、目标标签、规则基线或组合哈希。

V3.1联合任务：

1. `derived_risk_level`五分类；
2. `intervention_level`五分类；
3. `alert_mode`三分类；
4. `manual_review`二分类；
5. `abstain`二分类。

V3.1仍是离线监督训练。模型反馈必须经过人工审核后才能进入后续版本，不允许模型直接使用自己的预测进行在线自我训练。
