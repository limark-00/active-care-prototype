"""Partially fine-tune MacBERT together with the Event Transformer.

Only the last N MacBERT encoder layers are trainable.  The saved checkpoint is
a compact delta: it contains those trainable language-model parameters plus the
Event Transformer and output heads, not a second copy of the frozen base model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import TARGETS, dump_json, evaluate, load_jsonl
from .train_event_transformer import build_model, choose_device, split_segments

SEED = 20260831
SPLIT_NAMES = ("train", "validation", "test", "ood_test", "natural_test")
V31_TARGETS = {
    **TARGETS,
    "derived_risk_level": ["L0", "L1", "L2", "L3", "L4"],
}
SAFETY_SELECTION_WEIGHTS = {
    "intervention_level": 0.65,
    "derived_risk_level": 0.15,
    "alert_mode": 0.10,
    "manual_review": 0.05,
    "abstain": 0.05,
}
LOSS_WEIGHTS = {
    "intervention_level": 1.5,
    "derived_risk_level": 1.0,
    "alert_mode": 0.75,
    "manual_review": 0.5,
    "abstain": 0.5,
}


def find_encoder_layers(language_model):
    candidates = (
        getattr(getattr(language_model, "encoder", None), "layer", None),
        getattr(getattr(getattr(language_model, "bert", None), "encoder", None), "layer", None),
        getattr(getattr(getattr(language_model, "base_model", None), "encoder", None), "layer", None),
    )
    for layers in candidates:
        if layers is not None and len(layers):
            return layers
    raise ValueError("无法定位Transformer编码层；当前脚本仅支持BERT/MacBERT类模型结构")


def freeze_except_last_layers(language_model, unfreeze_layers: int) -> dict:
    layers = find_encoder_layers(language_model)
    if unfreeze_layers < 1 or unfreeze_layers > len(layers):
        raise ValueError(f"--unfreeze-layers需在1到{len(layers)}之间")
    for parameter in language_model.parameters():
        parameter.requires_grad = False
    for layer in layers[-unfreeze_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    total = sum(parameter.numel() for parameter in language_model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in language_model.parameters()
        if parameter.requires_grad
    )
    return {
        "encoder_layer_count": len(layers),
        "unfrozen_layer_count": unfreeze_layers,
        "language_model_parameter_count": total,
        "trainable_language_model_parameter_count": trainable,
    }


def build_joint_model(torch, language_model, d_model: int = 128, targets=TARGETS):
    nn = torch.nn
    task_model = build_model(
        torch, language_model.config.hidden_size, d_model=d_model, targets=targets
    )

    class JointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = language_model
            self.task_model = task_model

        def forward(self, input_ids, attention_mask, owners, slots, batch_size, slot_count):
            hidden = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            token_mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * token_mask).sum(1) / token_mask.sum(1).clamp_min(1)
            flat_positions = owners * slot_count + slots
            padded = pooled.new_zeros((batch_size * slot_count, pooled.shape[-1]))
            padded = padded.index_copy(0, flat_positions, pooled)
            event_vectors = padded.view(batch_size, slot_count, pooled.shape[-1])
            event_mask = torch.zeros(
                (batch_size, slot_count), dtype=torch.bool, device=pooled.device
            )
            event_mask[owners, slots] = True
            return self.task_model(event_vectors, event_mask)

    return JointModel()


def build_segment_table(tokenizer, splits: dict, max_length: int, torch):
    texts: list[str] = []
    vocabulary: dict[str, int] = {}
    row_segments: dict[str, list[list[int]]] = {}
    for split_name, rows in splits.items():
        packed_rows: list[list[int]] = []
        for row in rows:
            ids: list[int] = []
            for segment in split_segments(row["input_text"]):
                if segment not in vocabulary:
                    vocabulary[segment] = len(texts)
                    texts.append(segment)
                ids.append(vocabulary[segment])
            if not ids:
                raise ValueError(f"样本没有可编码文字段：{row.get('sample_id')}")
            packed_rows.append(ids)
        row_segments[split_name] = packed_rows
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"], row_segments, len(texts)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data/v3")
    parser.add_argument("--output-root", type=Path, default=root / "outputs")
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--task-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--aux-risk-head", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=("mean_macro_f1", "safety_weighted"),
        default="mean_macro_f1",
    )
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--underprediction-loss-weight", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("batch size与梯度累积步数必须大于0")
    if args.ordinal_loss_weight < 0 or args.underprediction_loss_weight < 0:
        raise ValueError("有序损失和欠干预损失权重不能为负数")
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = choose_device(torch, args.device)
    targets = V31_TARGETS if args.aux_risk_head else TARGETS
    total_started = time.time()
    run_prefix = "partial-macbert-v31-" if args.aux_risk_head else "partial-macbert-"
    run_dir = args.output_root / (
        run_prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True)

    splits = {
        name: load_jsonl(args.data_dir / f"{name}.jsonl") for name in SPLIT_NAMES
    }
    if args.train_limit:
        splits["train"] = splits["train"][: args.train_limit]
    if args.eval_limit:
        for name in SPLIT_NAMES[1:]:
            splits[name] = splits[name][: args.eval_limit]

    print(f"Loading {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    # PyTorch MPS currently rejects SDPA when attention dropout is active.
    # Eager attention keeps normal training dropout and works on CPU/MPS/CUDA.
    language_model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    freeze_info = freeze_except_last_layers(language_model, args.unfreeze_layers)
    model_commit = getattr(language_model.config, "_commit_hash", None)
    tokenization_started = time.time()
    input_ids_table, attention_table, segment_rows, unique_segments = build_segment_table(
        tokenizer, splits, args.max_length, torch
    )
    tokenization_seconds = time.time() - tokenization_started
    model = build_joint_model(torch, language_model, targets=targets).to(device)

    encoder_parameters = [
        parameter
        for parameter in model.language_model.parameters()
        if parameter.requires_grad
    ]
    task_parameters = list(model.task_model.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        (
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": task_parameters, "lr": args.task_lr},
        ),
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(splits["train"]) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = round(total_steps * args.warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    label_maps = {
        target: {str(value): index for index, value in enumerate(values)}
        for target, values in targets.items()
    }
    criterion = torch.nn.CrossEntropyLoss()

    def forward_rows(name: str, row_indices: list[int]):
        segments = [segment_rows[name][index] for index in row_indices]
        slot_count = max(len(values) for values in segments)
        flat_ids: list[int] = []
        owners: list[int] = []
        slots: list[int] = []
        for owner, values in enumerate(segments):
            for slot, segment_id in enumerate(values):
                flat_ids.append(segment_id)
                owners.append(owner)
                slots.append(slot)
        selected = torch.tensor(flat_ids, dtype=torch.long)
        return model(
            input_ids_table[selected].to(device),
            attention_table[selected].to(device),
            torch.tensor(owners, dtype=torch.long, device=device),
            torch.tensor(slots, dtype=torch.long, device=device),
            len(row_indices),
            slot_count,
        )

    def predict(name: str) -> dict:
        model.eval()
        collected = {target: [] for target in targets}
        with torch.inference_mode():
            for start in range(0, len(splits[name]), args.batch_size):
                indices = list(range(start, min(start + args.batch_size, len(splits[name]))))
                outputs = forward_rows(name, indices)
                for target, values in targets.items():
                    collected[target].extend(
                        values[index]
                        for index in outputs[target].argmax(1).cpu().tolist()
                    )
        return collected

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    def save_delta(path: Path) -> None:
        state = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if name in trainable_names
        }
        torch.save(
            {
                "state_dict": state,
                "base_model": args.model_name,
                "base_model_commit": model_commit,
                "unfreeze_layers": args.unfreeze_layers,
                "targets": targets,
                "d_model": 128,
                "max_length": args.max_length,
            },
            path,
        )

    best_score = -1.0
    stale_epochs = 0
    completed_epochs = 0
    global_step = 0
    training_started = time.time()
    log_path = run_dir / "train.jsonl"

    def selection_score(metrics: dict) -> float:
        if args.selection_metric == "mean_macro_f1":
            return metrics["mean_macro_f1"]
        available = {
            target: weight
            for target, weight in SAFETY_SELECTION_WEIGHTS.items()
            if target in targets
        }
        total_weight = sum(available.values())
        return sum(
            metrics[target]["macro_f1"] * weight
            for target, weight in available.items()
        ) / total_weight

    def batch_target(target: str, indices: list[int]):
        return torch.tensor(
            [
                label_maps[target][str(splits["train"][index][target])]
                for index in indices
            ],
            dtype=torch.long,
            device=device,
        )

    def training_loss(outputs: dict, indices: list[int]):
        total = torch.zeros((), device=device)
        for target, values in targets.items():
            truth = batch_target(target, indices)
            total = total + LOSS_WEIGHTS.get(target, 1.0) * criterion(
                outputs[target], truth
            )
            if target in {"intervention_level", "derived_risk_level"}:
                probabilities = outputs[target].softmax(dim=1)
                positions = torch.arange(
                    len(values), dtype=probabilities.dtype, device=device
                )
                expected = (probabilities * positions).sum(dim=1)
                truth_float = truth.to(probabilities.dtype)
                if args.ordinal_loss_weight:
                    total = total + args.ordinal_loss_weight * torch.nn.functional.smooth_l1_loss(
                        expected, truth_float
                    )
                if args.underprediction_loss_weight:
                    total = total + args.underprediction_loss_weight * torch.relu(
                        truth_float - expected
                    ).mean()
        return total

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(splits["train"]), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for batch_number, start in enumerate(
                range(0, len(order), args.batch_size), 1
            ):
                indices = order[start : start + args.batch_size]
                outputs = forward_rows("train", indices)
                loss = training_loss(outputs, indices)
                total_loss += loss.item() * len(indices)
                (loss / args.gradient_accumulation).backward()
                should_step = (
                    batch_number % args.gradient_accumulation == 0
                    or start + args.batch_size >= len(order)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        [*encoder_parameters, *task_parameters], 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if args.log_every and global_step % args.log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "status": "training",
                                    "epoch": epoch,
                                    "optimizer_step": global_step,
                                    "planned_optimizer_steps": total_steps,
                                    "latest_batch_loss": loss.item(),
                                    "elapsed_training_seconds": time.time()
                                    - training_started,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )

            validation_predictions = predict("validation")
            validation_metrics = evaluate(
                splits["validation"], validation_predictions, targets=targets
            )
            score = selection_score(validation_metrics)
            completed_epochs = epoch
            entry = {
                "epoch": epoch,
                "train_loss": total_loss / len(order),
                "validation_mean_macro_f1": validation_metrics["mean_macro_f1"],
                "validation_selection_score": score,
                "validation_mean_observed_macro_f1": validation_metrics[
                    "mean_observed_macro_f1"
                ],
                "validation_target_macro_f1": {
                    target: validation_metrics[target]["macro_f1"]
                    for target in targets
                },
                "validation_intervention_ordinal": validation_metrics[
                    "intervention_level"
                ].get("ordinal"),
                "optimizer_steps": global_step,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "task_lr": optimizer.param_groups[1]["lr"],
                "elapsed_training_seconds": time.time() - training_started,
            }
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(entry, ensure_ascii=False))
            if score > best_score + 1e-4:
                best_score = score
                stale_epochs = 0
                save_delta(run_dir / "best_delta.pt")
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    checkpoint = torch.load(
        run_dir / "best_delta.pt", map_location=device, weights_only=True
    )
    _, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint contains unexpected keys: {unexpected}")
    reports = {}
    for name in SPLIT_NAMES:
        reports[name] = evaluate(splits[name], predict(name), targets=targets)
        dump_json(run_dir / f"metrics-{name}.json", reports[name])

    training_seconds = time.time() - training_started
    run_record = {
        "model": "partial-macbert-event-transformer",
        "base_model": args.model_name,
        "base_model_commit": model_commit,
        "attention_implementation": "eager",
        "device": str(device),
        "data_dir": str(args.data_dir),
        "seed": SEED,
        **freeze_info,
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "task_parameter_count": sum(parameter.numel() for parameter in task_parameters),
        "unique_segments": unique_segments,
        "best_validation_score": best_score,
        "selection_metric": args.selection_metric,
        "epochs_completed": completed_epochs,
        "optimizer_steps": global_step,
        "total_seconds": time.time() - total_started,
        "tokenization_seconds": tokenization_seconds,
        "training_and_evaluation_seconds": training_seconds,
        "configuration": {
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "encoder_lr": args.encoder_lr,
            "task_lr": args.task_lr,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "patience": args.patience,
            "aux_risk_head": args.aux_risk_head,
            "ordinal_loss_weight": args.ordinal_loss_weight,
            "underprediction_loss_weight": args.underprediction_loss_weight,
            "loss_weights": {
                target: LOSS_WEIGHTS.get(target, 1.0) for target in targets
            },
            "selection_weights": (
                SAFETY_SELECTION_WEIGHTS
                if args.selection_metric == "safety_weighted"
                else None
            ),
        },
        "reports": {
            name: {
                "mean_macro_f1": report["mean_macro_f1"],
                "mean_observed_macro_f1": report["mean_observed_macro_f1"],
                "intervention_macro_f1": report["intervention_level"]["macro_f1"],
                "intervention_accuracy": report["intervention_level"]["accuracy"],
                "intervention_ordinal": report["intervention_level"].get("ordinal"),
                "risk_macro_f1": (
                    report["derived_risk_level"]["macro_f1"]
                    if "derived_risk_level" in report
                    else None
                ),
            }
            for name, report in reports.items()
        },
    }
    dump_json(run_dir / "run.json", run_record)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "best_validation_score": best_score,
                "scores": run_record["reports"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
