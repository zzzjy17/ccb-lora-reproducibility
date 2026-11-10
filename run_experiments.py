import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PROTOCOL_VERSION = "v2_independent_calibration_2026-10-27"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []
    buffer = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() and not buffer:
            continue
        buffer += line
        try:
            rows.append(json.loads(buffer))
            buffer = ""
        except json.JSONDecodeError:
            continue
    if buffer.strip():
        raise ValueError(f"Unparsed JSON content remains in {path}")
    return rows


def load_records(dataset):
    if dataset == "TNEWS":
        data = ROOT / "data" / "tnews"
        label_rows = [json.loads(x) for x in (data / "labels.json").read_text(encoding="utf-8").splitlines()]
        mapping = {int(r["label"]): i for i, r in enumerate(label_rows)}
        train = [json.loads(x) for x in (data / "train.json").read_text(encoding="utf-8").splitlines() if x.strip()]
        valid = [json.loads(x) for x in (data / "dev.json").read_text(encoding="utf-8").splitlines() if x.strip()]
        train_text = [r["sentence"] for r in train]
        train_y = [mapping[int(r["label"])] for r in train]
        eval_text = [r["sentence"] for r in valid]
        eval_y = [mapping[int(r["label"])] for r in valid]
        return train_text, np.array(train_y), eval_text, np.array(eval_y), len(mapping)
    if dataset == "ChnSentiCorp":
        data = ROOT / "data" / "chnsenticorp"
        train = read_jsonl(data / "train.jsonl")
        valid = read_jsonl(data / "validation.jsonl")
        test = read_jsonl(data / "test.jsonl")
        return ([r["text"] for r in train], np.array([r["label"] for r in train]),
                [r["text"] for r in test], np.array([r["label"] for r in test]), 2,
                [r["text"] for r in valid], np.array([r["label"] for r in valid]))
    if dataset == "IFLYTEK":
        data = ROOT / "data" / "iflytek"
        train = read_jsonl(data / "train.json")
        valid = read_jsonl(data / "dev.json")
        label_ids = sorted({int(r["label"]) for r in train + valid})
        mapping = {label: i for i, label in enumerate(label_ids)}
        return ([r["sentence"] for r in train], np.array([mapping[int(r["label"])] for r in train]),
                [r["sentence"] for r in valid], np.array([mapping[int(r["label"])] for r in valid]), len(mapping),
                [r["sentence"] for r in valid], np.array([mapping[int(r["label"])] for r in valid]))
    raise ValueError(dataset)


class DynamicLowRankHead(nn.Module):
    """Shared low-rank logits plus an uncertainty-gated class branch."""

    def __init__(self, hidden, num_labels, rank=4, gate_mode="uncertainty"):
        super().__init__()
        self.gate_mode = gate_mode
        self.base = nn.Linear(hidden, num_labels)
        self.a_shared = nn.Parameter(torch.randn(hidden, rank) * 0.02)
        self.b_shared = nn.Parameter(torch.randn(rank, num_labels) * 0.02)
        self.a_class = nn.Parameter(torch.randn(num_labels, hidden, rank) * 0.02)
        self.b_class = nn.Parameter(torch.randn(num_labels, rank) * 0.02)

    def forward(self, h, temperature=1.0):
        shared = self.base(h) + h @ self.a_shared @ self.b_shared
        calibrated = torch.softmax(shared / max(float(temperature), 1e-4), dim=-1)
        confidence = calibrated.max(dim=-1).values.detach()
        if self.gate_mode == "uncertainty":
            gate = 1.0 - confidence
        elif self.gate_mode == "confidence":
            gate = confidence
        elif self.gate_mode == "constant":
            gate = torch.ones_like(confidence)
        else:
            raise ValueError(f"Unknown gate mode: {self.gate_mode}")
        uncertainty = gate.unsqueeze(-1)
        class_delta = torch.einsum("bh,khr,kr->bk", h, self.a_class, self.b_class)
        return shared + uncertainty * class_delta, confidence


class BertLoRAClassifier(nn.Module):
    def __init__(self, model_name, num_labels, method, temperature=1.0, dcr_rank=4, dcr_gate="uncertainty"):
        super().__init__()
        self.method = method
        self.temperature = temperature
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        if method not in {"full_finetuning", "full_finetuning_ccb"}:
            cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                target_modules=["query", "value"],
                task_type="FEATURE_EXTRACTION",
            )
            self.encoder = get_peft_model(self.encoder, cfg)
        self.head = (DynamicLowRankHead(hidden, num_labels, rank=dcr_rank, gate_mode=dcr_gate)
                     if method == "dcr_ccb_lora" else nn.Linear(hidden, num_labels))

    def set_temperature(self, temperature):
        self.temperature = float(temperature)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        h = out.last_hidden_state[:, 0]
        if self.method == "dcr_ccb_lora":
            logits, conf = self.head(h, self.temperature)
            return logits, conf
        return self.head(h), None


def encode_batch(tokenizer, texts, device, max_length=128):
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def logits_for(model, tokenizer, texts, batch_size, train_mode=False):
    old = model.training
    model.train(train_mode)
    all_logits, all_conf = [], []
    with torch.no_grad():
        for st in range(0, len(texts), batch_size):
            batch = encode_batch(tokenizer, texts[st:st + batch_size], DEVICE)
            logits, conf = model(**batch)
            all_logits.append(logits.detach().cpu())
            if conf is not None:
                all_conf.append(conf.detach().cpu())
    model.train(old)
    return torch.cat(all_logits).numpy(), (torch.cat(all_conf).numpy() if all_conf else None)


def fit_temperature(logits, labels):
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / log_t.exp().clamp(0.05, 20.0), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach().clamp(0.05, 20.0).item())


def split_labeled_unlabeled(indices, labels, train_size, seed):
    """Stratify when possible; preserve singleton classes otherwise."""
    counts = np.bincount(np.asarray(labels, dtype=int))
    if len(counts) == 0 or counts.min() >= 2:
        return train_test_split(indices, train_size=train_size, stratify=labels, random_state=seed)
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    rare_classes = np.flatnonzero(counts < 2)
    mandatory = indices[np.isin(labels, rare_classes)]
    remaining = np.setdiff1d(indices, mandatory, assume_unique=False)
    need = int(train_size) - len(mandatory)
    if need < 0 or need > len(remaining):
        raise ValueError("train_size cannot preserve all singleton classes")
    chosen = np.concatenate([mandatory, rng.choice(remaining, size=need, replace=False)])
    return np.sort(chosen), np.setdiff1d(indices, chosen, assume_unique=False)


def ece(y, p, bins=10):
    conf = p.max(1)
    pred = p.argmax(1)
    out = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(float(conf[m].mean()) - float((pred[m] == y[m]).mean()))
    return float(out)


def metric_dict(y, logits, temperature=1.0):
    z = logits / max(temperature, 1e-4)
    p = torch.softmax(torch.tensor(z), dim=-1).numpy()
    pred = p.argmax(1)
    n = len(y)
    onehot = np.eye(p.shape[1])[y]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "ece": ece(y, p),
        "brier_score": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "nll": float(-np.log(np.clip(p[np.arange(n), y], 1e-8, 1)).mean()),
        "predictions": pred.tolist(),
        "probabilities": p.tolist(),
    }


def train_model(model, tokenizer, texts, labels, weights, epochs, batch_size, lr, head_lr=5e-4):
    model.to(DEVICE)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    trainable = encoder_params + head_params
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    opt = torch.optim.AdamW(groups)
    order = np.arange(len(texts))
    model.train()
    for _ in range(epochs):
        np.random.shuffle(order)
        for st in range(0, len(order), batch_size):
            ix = order[st:st + batch_size]
            batch = encode_batch(tokenizer, [texts[i] for i in ix], DEVICE)
            y = torch.tensor([int(labels[i]) for i in ix], dtype=torch.long, device=DEVICE)
            w = torch.tensor([weights[i] for i in ix], dtype=torch.float32, device=DEVICE)
            logits, _ = model(**batch)
            loss = (F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()


def select_pseudo(
    model,
    tokenizer,
    all_texts,
    unlabeled,
    thresholds,
    max_total,
    batch_size,
    consistency_passes=3,
    semantic_similarity_threshold=None,
):
    texts = [all_texts[i] for i in unlabeled]
    base_logits, _ = logits_for(model, tokenizer, texts, batch_size, train_mode=False)
    probs = torch.softmax(torch.tensor(base_logits / model.temperature), dim=-1).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    sortp = np.sort(probs, axis=1)
    margin = sortp[:, -1] - sortp[:, -2]
    stochastic = []
    for _ in range(consistency_passes):
        z, _ = logits_for(model, tokenizer, texts, batch_size, train_mode=True)
        stochastic.append(torch.softmax(torch.tensor(z / model.temperature), dim=-1).numpy())
    stack = np.stack(stochastic)
    consistent = (stack.argmax(2) == pred[None, :]).mean(0)
    std = stack.max(2).std(0)
    selected = []
    semantic_selected = []
    tfidf = None
    if semantic_similarity_threshold is not None:
        # Fit only on the candidate pool. L2-normalization makes the sparse
        # dot product equal cosine similarity, so the test remains cheap and
        # deterministic while preserving the audit trail.
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            max_features=10000,
            sublinear_tf=True,
            norm="l2",
        )
        tfidf = vectorizer.fit_transform(texts)
    counts = {int(c): 0 for c in range(probs.shape[1])}
    per_class_budget = max(1, int(max_total / probs.shape[1]))
    audit = []
    for j, original_idx in enumerate(unlabeled):
        threshold = thresholds.get(int(pred[j]), 0.90)
        reason = "selected"
        max_similarity = 0.0
        was_selected = False
        ok = conf[j] >= threshold and margin[j] >= 0.03 and consistent[j] >= 0.60 and std[j] <= 0.20
        if counts[int(pred[j])] >= per_class_budget:
            ok = False
            reason = "class_budget"
        elif conf[j] < threshold:
            reason = "confidence"
        elif margin[j] < 0.03:
            reason = "margin"
        elif consistent[j] < 0.60:
            reason = "consistency"
        elif std[j] > 0.20:
            reason = "dispersion"
        if ok and semantic_similarity_threshold is not None and semantic_selected:
            max_similarity = float(
                tfidf[j].dot(tfidf[semantic_selected].T).toarray().max()
            )
            if max_similarity > semantic_similarity_threshold:
                ok = False
                reason = "semantic_similarity"
        if ok and len(selected) < max_total:
            selected.append((original_idx, int(pred[j]), float(conf[j])))
            counts[int(pred[j])] += 1
            semantic_selected.append(j)
            was_selected = True
        audit.append({
            "index": int(original_idx),
            "predicted_class": int(pred[j]),
            "calibrated_confidence": float(conf[j]),
            "margin": float(margin[j]),
            "consistency": float(consistent[j]),
            "probability_std": float(std[j]),
            "selected": bool(was_selected),
            "reason": reason,
            "max_selected_cosine": float(max_similarity),
        })
    return selected, counts, audit


def run_one(args, dataset, fraction, seed, method, tokenizer):
    seed_everything(seed)
    loaded = load_records(dataset)
    calibration_source = "dataset_validation_split"
    calibration_indices = []
    evaluation_indices = []
    if dataset in {"ChnSentiCorp", "IFLYTEK"}:
        train_text, train_y, eval_text, eval_y, num_labels, cal_text, cal_y = loaded
        if dataset == "IFLYTEK":
            # IFLYTEK's official test split is unlabeled in the local release.
            # Split the labeled dev set deterministically per seed so calibration
            # and evaluation remain disjoint.
            cal_idx, eval_idx = train_test_split(
                np.arange(len(eval_text)), test_size=0.5, random_state=seed
            )
            cal_text = [eval_text[i] for i in cal_idx]
            cal_y = eval_y[cal_idx]
            eval_text = [eval_text[i] for i in eval_idx]
            eval_y = eval_y[eval_idx]
            calibration_source = "IFLYTEK_dev_half_by_seed"
            calibration_indices = [int(i) for i in cal_idx]
            evaluation_indices = [int(i) for i in eval_idx]
    else:
        train_text, train_y, eval_text, eval_y, num_labels = loaded
        # TNEWS has one labeled dev split in the local release. Keep a
        # disjoint half for temperature fitting and the other half for final
        # evaluation; evaluation labels never enter calibration.
        cal_idx, eval_idx = train_test_split(
            np.arange(len(eval_text)), test_size=0.5, random_state=seed,
            stratify=eval_y,
        )
        cal_text = [eval_text[i] for i in cal_idx]
        cal_y = eval_y[cal_idx]
        eval_text = [eval_text[i] for i in eval_idx]
        eval_y = eval_y[eval_idx]
        calibration_source = "TNEWS_dev_half_by_seed"
        calibration_indices = [int(i) for i in cal_idx]
        evaluation_indices = [int(i) for i in eval_idx]
    if dataset == "ChnSentiCorp":
        calibration_indices = list(range(len(cal_text)))
        evaluation_indices = list(range(len(eval_text)))
    source_train_indices = np.arange(len(train_text), dtype=int)
    if args.max_train_samples and len(train_text) > args.max_train_samples:
        keep, _ = split_labeled_unlabeled(np.arange(len(train_text)), train_y, args.max_train_samples, seed)
        keep = np.sort(keep)
        train_text = [train_text[i] for i in keep]
        train_y = train_y[keep]
        source_train_indices = source_train_indices[keep]
    indices = np.arange(len(train_text))
    labeled, unlabeled = split_labeled_unlabeled(indices, train_y, int(len(indices) * fraction), seed)
    if args.max_unlabeled and len(unlabeled) > args.max_unlabeled:
        rng = np.random.default_rng(seed)
        unlabeled = np.sort(rng.choice(unlabeled, args.max_unlabeled, replace=False))
    model_name = args.model_name
    model = BertLoRAClassifier(model_name, num_labels, method, dcr_rank=args.dcr_rank, dcr_gate=args.dcr_gate)
    batch = args.batch_size
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_dir = ROOT / "results" / "runs" / f"{dataset}_{fraction:.2f}_{seed}_{method}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_model(model, tokenizer, [train_text[i] for i in labeled], train_y[labeled], np.ones(len(labeled)), args.epochs, batch, args.lr)
    cal_logits, _ = logits_for(model, tokenizer, cal_text, batch)
    calibrated = method not in {"rc_no_calibration", "dcr_no_calibration", "lora", "full_finetuning"}
    temperature = fit_temperature(cal_logits, cal_y) if calibrated else 1.0
    model.set_temperature(temperature)
    pseudo = []
    audit = []
    use_pseudo = method in {
        "rc_ccb_lora", "dcr_ccb_lora", "dcr_no_calibration", "rc_no_calibration",
        "full_finetuning_ccb",
    }
    if method == "dcr_no_pseudo":
        # Give the dynamic head a supervised adaptation stage after the
        # calibration temperature has been estimated, so the ablation tests
        # the structure itself rather than an under-trained random head.
        train_model(model, tokenizer, [train_text[i] for i in labeled], train_y[labeled], np.ones(len(labeled)), args.epochs, batch, args.lr)
    if use_pseudo:
        ul = unlabeled
        if args.max_unlabeled is None:
            ul = unlabeled
        cal_p = torch.softmax(torch.tensor(cal_logits / temperature), dim=-1).numpy()
        cal_pred = cal_p.argmax(1)
        thresholds = {}
        for c in range(num_labels):
            vals = cal_p[cal_pred == c, c]
            thresholds[c] = float(np.clip(np.quantile(vals, 0.70) if len(vals) else 0.90, 0.75, 0.95))
        pseudo_budget = min(2000, max(1, int(len(ul) * args.pseudo_budget_ratio)))
        pseudo, counts, audit = select_pseudo(
            model,
            tokenizer,
            train_text,
            ul,
            thresholds,
            pseudo_budget,
            batch,
            args.consistency_passes,
            args.semantic_similarity_threshold,
        )
        for row in audit:
            row["source_index"] = int(source_train_indices[int(row["index"])])
        if pseudo:
            pseudo_text = [train_text[i] for i, _, _ in pseudo]
            pseudo_y = np.array([y for _, y, _ in pseudo])
            text2 = [train_text[i] for i in labeled] + pseudo_text
            y2 = np.concatenate([train_y[labeled], pseudo_y])
            w2 = np.concatenate([np.ones(len(labeled)), np.full(len(pseudo), 0.5)])
            train_model(model, tokenizer, text2, y2, w2, args.epochs, batch, args.lr)
    else:
        counts = {}
        thresholds = {}
    logits, _ = logits_for(model, tokenizer, eval_text, batch)
    metrics = metric_dict(eval_y, logits, model.temperature)
    pred = metrics.pop("predictions")
    probs = metrics.pop("probabilities")
    result = {
        "dataset": dataset,
        "label_fraction": fraction,
        "seed": seed,
        "method": method,
        "model_name": model_name,
        "num_labels": num_labels,
        "n_train": len(train_text),
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "n_eval": len(eval_text),
        "epochs": args.epochs,
        "protocol_version": args.protocol_version,
        "code_sha256": CODE_SHA256,
        "calibration_source": calibration_source,
        "calibration_size": len(cal_text),
        "evaluation_size": len(eval_text),
        "temperature": temperature,
        "dcr_rank": args.dcr_rank,
        "dcr_gate": args.dcr_gate,
        "pseudo_label_count": len(pseudo),
        "pseudo_label_budget": min(2000, max(1, int(len(unlabeled) * args.pseudo_budget_ratio))),
        "pseudo_budget_ratio": args.pseudo_budget_ratio,
        "semantic_similarity_threshold": args.semantic_similarity_threshold,
        "pseudo_class_counts": counts,
        "thresholds": thresholds,
        "train_seconds": time.time() - t0,
        "device": str(DEVICE),
        "pilot_caps": {"max_train_samples": args.max_train_samples, "max_unlabeled": args.max_unlabeled},
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "predictions.npz", y_true=eval_y, y_pred=np.array(pred), probabilities=np.array(probs))
    if audit:
        with (run_dir / "pseudo_label_audit.jsonl").open("w", encoding="utf-8") as f:
            for row in audit:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps({
        "dataset": dataset, "fraction": fraction, "seed": seed, "method": method,
        "labeled_indices": labeled.tolist(), "unlabeled_indices": unlabeled.tolist(),
        "source_train_indices": source_train_indices.tolist(),
        "calibration_indices": calibration_indices,
        "evaluation_indices": evaluation_indices,
        "calibration_source": calibration_source,
        "temperature": temperature,
        "protocol_version": args.protocol_version,
        "code_sha256": CODE_SHA256,
        "semantic_similarity_threshold": args.semantic_similarity_threshold,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ChnSentiCorp"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.10])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--methods", nargs="+", default=["lora", "rc_ccb_lora", "dcr_ccb_lora"])
    ap.add_argument("--model-name", default="google-bert/bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--consistency-passes", type=int, default=3)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-unlabeled", type=int, default=None)
    ap.add_argument("--dcr-rank", type=int, default=4)
    ap.add_argument("--dcr-gate", choices=["uncertainty", "confidence", "constant"], default="uncertainty")
    ap.add_argument("--pseudo-budget-ratio", type=float, default=0.05)
    ap.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=None,
        help="Optional cosine threshold for greedy TF-IDF semantic diversity filtering.",
    )
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    all_results = []
    for dataset in args.datasets:
        for fraction in args.fractions:
            for seed in args.seeds:
                for method in args.methods:
                    tag = f"_{args.run_tag}" if args.run_tag else ""
                    path = ROOT / "results" / "runs" / f"{dataset}_{fraction:.2f}_{seed}_{method}{tag}" / "metrics.json"
                    if path.exists():
                        all_results.append(json.loads(path.read_text(encoding="utf-8")))
                        continue
                    print(f"RUN {dataset} frac={fraction} seed={seed} method={method} device={DEVICE}", flush=True)
                    all_results.append(run_one(args, dataset, fraction, seed, method, tokenizer))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    fields = sorted({k for r in all_results for k, v in r.items() if not isinstance(v, (dict, list))})
    with (out / "all_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in fields})
    import pandas as pd
    df = pd.DataFrame(all_results)
    if not df.empty:
        summary = df.groupby(["dataset", "label_fraction", "method"])[["accuracy", "macro_f1", "weighted_f1", "ece", "brier_score", "nll", "pseudo_label_count"]].agg(["mean", "std", "count"]).reset_index()
        summary.to_csv(out / "summary.csv", index=False)
    print(json.dumps({"runs": len(all_results), "results": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
