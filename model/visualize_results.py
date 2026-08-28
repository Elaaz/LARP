

import os, json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPLITS = ["test_seen", "test_unseen_objects", "test_unseen_configs"]
SPLIT_LABELS = ["Seen", "Unseen objects", "Unseen configs"]

C_GAT, C_SYM, C_ORA = "#1f66a1", "#be2e2e", "#9aa0a6"                  


def fig_f1_comparison(R, out):
    gat = [R[s]["gat"]["F1"] for s in SPLITS]
    sym = [R[s]["symbolic_observed"]["F1"] for s in SPLITS]
    ora = [R[s]["symbolic_oracle"]["F1"] for s in SPLITS]
    x = np.arange(len(SPLITS)); w = 0.26

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(x - w, gat, w, label="GAT (noisy obs.)", color=C_GAT)
    b2 = ax.bar(x,     sym, w, label="Symbolic (noisy obs.)", color=C_SYM)
    b3 = ax.bar(x + w, ora, w, label="Symbolic (oracle)", color=C_ORA, hatch="//",
                edgecolor="white")
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(SPLIT_LABELS)
    ax.set_ylabel("Detection F1"); ax.set_ylim(0, 1.12)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Missing-step detection under perception noise", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_f1_comparison.{ext}"), dpi=300)
    plt.close(fig)


def fig_gat_heads(R, out):
    f1  = [R[s]["gat"]["F1"] for s in SPLITS]
    blk = [R[s]["gat"]["blk_acc"] for s in SPLITS]
    cat = [R[s]["gat"]["cat_acc"] for s in SPLITS]
    x = np.arange(len(SPLITS)); w = 0.26

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(x - w, f1, w, label="Detection F1", color=C_GAT)
    b2 = ax.bar(x,     blk, w, label="Blocking-object acc.", color="#1b8035")
    b3 = ax.bar(x + w, cat, w, label="Category acc.", color="#785aa0")
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(SPLIT_LABELS)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("GAT verifier output heads by split", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_gat_heads.{ext}"), dpi=300)
    plt.close(fig)


def fig_gap(R, out):
    gaps = [R[s]["gat"]["F1"] - R[s]["symbolic_observed"]["F1"] for s in SPLITS]
    colors = [C_GAT if g >= 0 else C_SYM for g in gaps]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(SPLIT_LABELS, gaps, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%+.3f", fontsize=9, padding=3)
    ax.axhline(0, color="black", lw=0.8)
    lo, hi = min(gaps), max(gaps)
    ax.set_ylim(min(lo * 1.45, -0.02), max(hi * 1.3, 0.02))
    ax.set_ylabel("F1 gap  (GAT − symbolic, noisy obs.)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Robustness gap under perception noise", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_gap.{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/results.json")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = json.load(open(args.results))
    fig_f1_comparison(R, args.out)
    fig_gat_heads(R, args.out)
    fig_gap(R, args.out)
    print(f"[+] wrote 3 figures (png+pdf) to {args.out}/")


if __name__ == "__main__":
    main()
