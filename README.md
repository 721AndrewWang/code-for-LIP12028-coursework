# code‑for‑LIP12028‑coursework

Source code for **LIP12028‑202425 – Datasets & Natural Language Processing** coursework (final paper: *“Interpretable Multilingual Crisis Classification via Model Compression”*).

---

## Contents

| File / Dir          | Purpose                                                                         |
| ------------------- | ------------------------------------------------------------------------------- |
| `train_compress.py` | End‑to‑end pipeline: KD ⇒ structured pruning ⇒ INT8 quantisation + latency test |
| `requirements.txt`  | Minimal Python dependencies (PyTorch 2.2, Transformers 4.40, Intel INC etc.)    |
| `.gitignore`        | Excludes checkpoints (`*.pt`, `outputs/`) & data (`data/`) from VCS             |

---

## Quick start

```bash
# 1 Install deps (CPU‑only); use CUDA wheel if you have a GPU
pip install -r requirements.txt

# 2 Run a short sanity‑check (1 epoch, small batch)
python train_compress.py \
    --epochs 1 --batch_size 4 --data_dir data
```

Full training (3 epochs, batch 32) writes three checkpoints to `outputs/`:

* `student_fp32/` – distilled 6‑layer model
* `student_pruned/` – after 30 % Ln‑structured pruning
* `student_int8/` – static INT8 model (CPU‑ready)

---

## Reproducing paper results

1. Place **CrisisNLP** `crisis_train.jsonl` / `crisis_dev.jsonl` in `data/` (each line: `{ "text": ..., "label": int }`).
2. Run the default settings (3 epochs).
3. Evaluate macro‑F1 on the dev set – should match numbers in Table 2 of the paper.

---

## License

MIT License – see `LICENSE` (feel free to reuse with citation).

---

> *For any issues please open a discussion or contact the author (wangj29@tcd.ie).*
