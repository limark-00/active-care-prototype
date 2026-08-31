"""Local V3.1 text-decision inference; no network API or input persistence."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_model.training.train_event_transformer import choose_device, split_segments
from decision_model.training.train_partial_macbert import V31_TARGETS, build_joint_model


class DecisionModelUnavailable(RuntimeError):
    pass


def latest_checkpoint() -> Path:
    explicit = os.environ.get("CARE_DECISION_CHECKPOINT")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_dir():
            path = path / "best_delta.pt"
        if not path.is_file():
            raise DecisionModelUnavailable(f"指定的决策模型不存在：{path}")
        return path
    outputs = PROJECT_ROOT / "decision_model" / "outputs"
    candidates = sorted(
        outputs.glob("partial-macbert-v31-*/best_delta.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise DecisionModelUnavailable(
            "未找到V3.1模型，请先运行 decision_model/run_partial_finetuning_v31.command。"
        )
    return candidates[-1]


def probability_payload(values, probabilities) -> dict[str, float]:
    return {
        str(value).lower() if isinstance(value, bool) else str(value): round(float(probability), 6)
        for value, probability in zip(values, probabilities)
    }


def apply_decision_guardrails(predictions: dict) -> tuple[dict, list[str]]:
    """Apply presentation invariants without changing the five raw heads."""
    risk = str(predictions["derived_risk_level"]["label"])
    intervention = str(predictions["intervention_level"]["label"])
    alert = str(predictions["alert_mode"]["label"])
    manual = bool(predictions["manual_review"]["label"])
    abstain = bool(predictions["abstain"]["label"])
    reasons: list[str] = []
    low_confidence = [
        target
        for target in (
            "derived_risk_level",
            "intervention_level",
            "alert_mode",
            "manual_review",
            "abstain",
        )
        if float(predictions[target]["confidence"]) < 0.55
    ]
    if low_confidence:
        manual = True
        reasons.append("存在低置信度任务输出，转人工核查")
    if risk == "L4" and intervention != "I4":
        intervention = "I4"
        alert = "URGENT_HELP"
        manual = True
        reasons.append("L4与干预输出不一致，按合成策略不变量升级为I4")
    if intervention == "I4" and alert != "URGENT_HELP":
        alert = "URGENT_HELP"
        manual = True
        reasons.append("I4与警示输出不一致，升级为紧急网页求助")
    if risk in {"L2", "L3"} and alert == "NONE":
        alert = "PAGE_WARNING"
        manual = True
        reasons.append("L2/L3与无警示输出不一致，增加网页警告")
    if abstain:
        manual = True
        reasons.append("模型选择拒绝自动判断，等待人工核查")
    if bool(predictions["manual_review"]["label"]):
        reasons.append("模型预测需要人工核查")
    return (
        {
            "risk_level": risk,
            "intervention_level": intervention,
            "alert_mode": alert,
            "manual_review": manual,
            "abstain": abstain,
        },
        list(dict.fromkeys(reasons)),
    )


class TextDecisionModel:
    model_name = "MacBERT + Event Transformer V3.1"

    def __init__(self, checkpoint_path: Path | None = None, device_name: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        self.torch = torch
        self.checkpoint_path = checkpoint_path or latest_checkpoint()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.targets = checkpoint["targets"]
        if self.targets != V31_TARGETS:
            raise DecisionModelUnavailable("检查点的V3.1任务标签或顺序不兼容。")
        requested_device = device_name or os.environ.get("CARE_DECISION_DEVICE", "auto")
        self.device = choose_device(torch, requested_device)
        self.max_length = int(checkpoint.get("max_length", 192))
        base_model = checkpoint["base_model"]
        base_revision = checkpoint.get("base_model_commit")
        pretrained_options = {
            "trust_remote_code": False,
            "local_files_only": True,
        }
        if base_revision:
            pretrained_options["revision"] = base_revision
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, **pretrained_options)
        language_model = AutoModel.from_pretrained(
            base_model,
            **pretrained_options,
            attn_implementation="eager",
        )
        self.model = build_joint_model(
            torch,
            language_model,
            d_model=int(checkpoint.get("d_model", 128)),
            targets=self.targets,
        )
        delta = checkpoint.get("state_dict", {})
        required_head_keys = {
            f"task_model.heads.{target}.{suffix}"
            for target in self.targets
            for suffix in ("weight", "bias")
        }
        if not required_head_keys.issubset(delta):
            raise DecisionModelUnavailable("模型检查点缺少一个或多个V3.1任务头。")
        _, unexpected = self.model.load_state_dict(delta, strict=False)
        if unexpected:
            raise DecisionModelUnavailable(f"模型检查点包含未知参数：{unexpected}")
        self.model.to(self.device).eval()
        self.lock = threading.Lock()
        self.run_name = self.checkpoint_path.parent.name
        self.base_model = base_model
        self.base_model_revision = base_revision

    def predict(self, text: str) -> dict:
        normalized = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
        if not 10 <= len(normalized) <= 6000:
            raise ValueError("文字长度需在10到6000个字符之间。")
        segments = split_segments(normalized)
        if not segments or not any(segment.strip() for segment in segments):
            raise ValueError("没有可用于判断的文字内容。")
        started = time.monotonic()
        with self.lock, self.torch.inference_mode():
            encoded = self.tokenizer(
                segments,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            count = len(segments)
            outputs = self.model(
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
                self.torch.zeros(count, dtype=self.torch.long, device=self.device),
                self.torch.arange(count, dtype=self.torch.long, device=self.device),
                1,
                count,
            )
            predictions = {}
            for target, values in self.targets.items():
                probabilities = outputs[target].softmax(dim=1)[0].detach().cpu()
                index = int(probabilities.argmax())
                predictions[target] = {
                    "label": values[index],
                    "confidence": round(float(probabilities[index]), 6),
                    "probabilities": probability_payload(values, probabilities),
                }

        guarded, reasons = apply_decision_guardrails(predictions)

        return {
            "schema_version": 1,
            "source": "local_text_model",
            "model": self.model_name,
            "base_model": self.base_model,
            "run_name": self.run_name,
            "device": str(self.device),
            "input_characters": len(normalized),
            "segment_count": len(segments),
            "inference_ms": round((time.monotonic() - started) * 1000, 1),
            "model_output": {
                "risk_level": predictions["derived_risk_level"],
                "intervention_level": predictions["intervention_level"],
                "alert_mode": predictions["alert_mode"],
                "manual_review": predictions["manual_review"],
                "abstain": predictions["abstain"],
            },
            "guarded_output": {
                **guarded,
            },
            "guardrail_applied": bool(reasons),
            "review_reasons": list(dict.fromkeys(reasons)),
            "limitations": "合成数据模型输出，仅用于原型演示；未连接真实求助或设备控制。",
        }


def decision_status() -> dict:
    try:
        checkpoint = latest_checkpoint()
    except DecisionModelUnavailable as exc:
        return {"available": False, "detail": str(exc)}
    return {"available": True, "run_name": checkpoint.parent.name}
