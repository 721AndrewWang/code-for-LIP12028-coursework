#!/usr/bin/env python
"""
train_compress.py
=================
End‑to‑end compression pipeline for the LIP12028‑202425 assignment
("Datasets and Natural Language Processing"). The script performs:

1. Load a 12‑layer XLM‑RoBERTa **teacher** model.
2. Build a 6‑layer **student** and fine‑tune it via knowledge distillation.
3. **Structured pruning**
   • attention‑head pruning (2 heads/layer).
   • Ln‑structured weight pruning (30 % of large Linear layers).
4. Static **INT8** post‑training quantisation (Intel Neural Compressor).
5. Save three checkpoints (FP32, pruned FP32, INT8).
6. Report CPU inference latency on a demo sentence.

Run on CPU‑only or GPU‑equipped machines; the script auto‑detects CUDA.

Dependencies
------------
```bash
pip install "torch>=2.2" "transformers>=4.40" datasets \
            "optimum[intel]" neural-compressor scikit-learn
```
"""

import argparse
import json
import os
import random
import time
from typing import Dict, Iterator

import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.nn.utils import prune
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from optimum.intel import INCQuantizer
from neural_compressor import PostTrainingQuantConfig
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1_metric(eval_pred):
    """Hugging Face Trainer metric callback: macro‑averaged F1."""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"macro_f1": f1_score(labels.astype("int32"), preds.astype("int32"), average="macro")}


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Cross‑entropy + KL divergence (soft‑target) loss."""
    ce = nn.functional.cross_entropy(student_logits, labels)
    kl = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=-1),
        nn.functional.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature ** 2
    return alpha * kl + (1.0 - alpha) * ce


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="xlm-roberta-base")
    parser.add_argument("--data_dir", default="data", help="directory with crisis_*.jsonl files")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed()

    # Detect device ----------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Teacher -------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model, num_labels=9
    ).to(device).eval()

    # 2. Dataset -------------------------------------------------------------

    def stream_jsonl(split: str) -> Iterator[Dict]:
        path = os.path.join(args.data_dir, f"crisis_{split}.jsonl")
        with open(path, encoding="utf‑8") as fp:
            for line in fp:
                rec = json.loads(line)
                yield {"text": rec["text"], "label": rec["label"]}

    def tokenize(rec: Dict) -> Dict:
        enc = tokenizer(
            rec["text"], truncation=True, padding=False, max_length=args.max_len
        )
        enc["labels"] = rec["label"]
        return enc

    train_ds = Dataset.from_list(list(stream_jsonl("train"))).map(tokenize, remove_columns=["text"])
    dev_ds = Dataset.from_list(list(stream_jsonl("dev"))).map(tokenize, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # 3. Student + KD --------------------------------------------------------
    config = AutoConfig.from_pretrained(args.teacher_model, num_labels=9)
    config.num_hidden_layers = 6
    student = AutoModelForSequenceClassification.from_config(config).to(device)

    # 3a. Attention‑head pruning (remove heads 0 & 11 per layer)
    heads_to_prune = {layer: {0, 11} for layer in range(6)}
    student.base_model.encoder.prune_heads(heads_to_prune)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_steps=50,
        seed=42,
        report_to=[],
        save_strategy="no",
    )

    def kd_loss_wrapper(model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        student_out = model(**inputs)
        with torch.no_grad():
            teacher_out = teacher(**inputs)
        loss = distillation_loss(student_out.logits, teacher_out.logits, labels)
        return (loss, student_out) if return_outputs else loss

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=macro_f1_metric,
        loss_func=kd_loss_wrapper,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "student_fp32"))

    # 4. Ln‑structured pruning (30 %) --------------------------------------
    for name, module in student.named_modules():
        if isinstance(module, nn.Linear) and module.out_features >= 512:
            prune.ln_structured(module, name="weight", amount=0.3, n=1, dim=0)
            prune.remove(module, "weight")

    trainer.train()  # fine‑tune after pruning
    trainer.save_model(os.path.join(args.output_dir, "student_pruned"))

    # 5. INT8 quantisation ---------------------------------------------------
    student.cpu()  # INC 仅支持 CPU 模型
    device = "cpu"

    calib_subset = dev_ds.shuffle(seed=42).select(range(512))

    def calib_loader(batch_size: int = 8):
        for i in range(0, len(calib_subset), batch_size):
            batch = calib_subset[i : i + batch_size].with_format("torch")
            yield {k: v for k, v in batch.items()}

    quantiser = INCQuantizer(
        model=student,
        quant_config=PostTrainingQuantConfig(approach="static"),
    )

    def inc_eval(model):
        res = trainer.predict(dev_ds, model=model)
        preds = res.predictions.argmax(-1)
        return f1_score(res.label_ids.astype("int32"), preds.astype("int32"), average="macro")

    int8_model = quantiser.fit(
        calib_dataloader=calib_loader(),
        eval_func=inc_eval,
    )
    int8_model.save_pretrained(os.path.join(args.output_dir, "student_int8"))

    # 6. CPU latency demo ----------------------------------------------------
    sample = "Need medical aid in Puebla!"
    enc = tokenizer(sample, return_tensors="pt")
    start = time.perf_counter()
    logits = int8_model(**enc).logits
    lat_ms = (time.perf_counter() - start) * 1000.0
    pred = logits.argmax(-1).item()
    print(f"CPU latency: {lat_ms:.1f} ms | predicted label = {pred}")


if __name__ == "__main__":
    main()
