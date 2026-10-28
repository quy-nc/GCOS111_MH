"""Bước 4b: biểu đồ bổ sung cho phần đánh giá. Không chọn lại model, không đọc test.csv
(trừ biểu đồ phân bố lớp, chỉ đếm nhãn).

Biểu đồ:
    figures/metrics_svm_vs_nb.png   Accuracy / Precision / Recall / F1 (macro, trung bình 8 aspect) của SVM và NB, CV 15 fold ± std
    figures/cv_folds.png            Macro-F1 của từng fold cho E0, E1, A1, A2, B1-SVM, B1-NB
    figures/top_words_svm.png       Từ (word TF-IDF) có trọng số lớn nhất của SVM cho lớp pos và non_pos, theo aspect
    figures/class_distribution.png  Phân bố none / pos / non_pos theo aspect, train và test
    figures/timing.png              Thời gian train và dự đoán mỗi fold của SVM và NB

Cấu hình tốt nhất của từng thí nghiệm lấy từ results/cv_results.csv (bước 3) rồi chạy lại CV trên đúng các fold đó,
tuần tự (n_jobs=1) để đo thời gian công bằng.

Input : data/absa/train.csv, data/absa/test.csv (chỉ đếm nhãn), results/cv_results.csv, results/baselines_cv.csv,
        models/pipeline.pkl
Output: figures/*.png, results/figures_cv.csv (metric + thời gian từng fold), results/top_words_svm.md

Chạy:
    python main.py figures
"""

import argparse
import importlib
import re
import sys
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from absa_common import ASPECTS, CLASSES3, LABEL_COLS, PROJECT_DIR, RESULTS_DIR, cv_folds, load_split  # noqa: E402
from clean_data import to_markdown  # noqa: E402

FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
make_pipe = importlib.import_module("03_train_cv").make_pipe

# Màu theo bảng mặc định của skill dataviz (đã chạy validate_palette.js, mode light)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLOR = {"SVM": "#2a78d6", "NB": "#eda100"}
CLASS_COLOR = {"none": "#c3c2b7", "pos": "#2a78d6", "non_pos": "#eb6834"}
HIGHLIGHT = {"B1-SVM": "#2a78d6", "E1_rule": "#eb6834"}  # cùng màu với figures/f1_by_aspect.png

PARAM_RE = re.compile(r"([\w.]+)=(\([^)]*\)|[^,]+)")
NB_CLASSES = {"MultinomialNB": MultinomialNB, "ComplementNB": ComplementNB}


def parse_value(v: str):
    v = v.strip()
    if v == "None":
        return None
    if v.startswith("("):
        return tuple(int(x) for x in v.strip("()").split(","))
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def build_best(experiment: str, params: str):
    """Dựng lại pipeline từ chuỗi tham số trong cv_results.csv (định dạng của describe() ở 03_train_cv.py)."""
    kv = {k: parse_value(v) for k, v in PARAM_RE.findall(params)}
    is_svm = "C" in kv
    pipe = make_pipe(LinearSVC(max_iter=10000) if is_svm else MultinomialNB(), experiment.startswith("B1"))
    if not is_svm:
        pipe.set_params(clf__estimator=NB_CLASSES[kv.pop("nb")]())
    mapping = {"C": "clf__estimator__C", "class_weight": "clf__estimator__class_weight",
               "alpha": "clf__estimator__alpha", "word.min_df": "feat__word__min_df",
               "word.ngram_range": "feat__word__ngram_range"}
    pipe.set_params(**{mapping[k]: v for k, v in kv.items()})
    return pipe


def fold_metrics(Y: np.ndarray, P: np.ndarray) -> dict:
    """Accuracy và Precision / Recall / F1 macro (3 lớp) cho từng aspect, rồi trung bình 8 aspect."""
    acc, pr, rc, f1 = [], [], [], []
    for j in range(Y.shape[1]):
        acc.append(accuracy_score(Y[:, j], P[:, j]))
        p, r, f, _ = precision_recall_fscore_support(Y[:, j], P[:, j], labels=CLASSES3, average="macro", zero_division=0)
        pr.append(p), rc.append(r), f1.append(f)
    return {"accuracy": np.mean(acc), "precision": np.mean(pr), "recall": np.mean(rc), "f1": np.mean(f1)}


def run_cv(configs: dict, X, Y, folds) -> pd.DataFrame:
    rows = []
    for name, pipe in configs.items():
        print(f"[{name}] chạy lại {len(folds)} fold ...", flush=True)
        for k, (tr, va) in enumerate(folds):
            model = clone(pipe)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X.iloc[tr], Y.iloc[tr])
            t1 = time.perf_counter()
            P = np.asarray(model.predict(X.iloc[va]))
            t2 = time.perf_counter()
            rows.append({"experiment": name, "fold": k, "fit_s": t1 - t0, "predict_s": t2 - t1,
                         "n_val": len(va), **fold_metrics(Y.values[va], P)})
    return pd.DataFrame(rows)


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left" if grid_axis == "y" else "bottom"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)
    (ax.yaxis if grid_axis == "y" else ax.xaxis).grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(cv: pd.DataFrame, svm: str, nb: str):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    style_axes(ax)
    x, w = np.arange(len(metrics)), 0.36
    for i, (exp, short) in enumerate([(svm, "SVM"), (nb, "NB")]):
        g = cv[cv.experiment == exp][metrics]
        mean, std = g.mean().values, g.std().values
        bars = ax.bar(x + (i - 0.5) * w, mean, w, yerr=std, color=MODEL_COLOR[short], edgecolor=SURFACE,
                      linewidth=1.5, label=f"{short} ({exp})", error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
        for b, m in zip(bars, mean):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, labels, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Trung bình 8 aspect", color=INK2)
    ax.set_title("SVM và Naive Bayes trên CV của train (15 fold, thanh = ± std)", color=INK, loc="left", fontsize=12, pad=26)
    ax.legend(frameon=False, ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK2, borderaxespad=0.2)
    save(fig, "metrics_svm_vs_nb.png")


def plot_cv_folds(scores: pd.DataFrame, order: list[str]):
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    style_axes(ax)
    rng = np.random.default_rng(0)
    for i, exp in enumerate(order):
        v = scores[scores.experiment == exp]["f1"].values
        color = HIGHLIGHT.get(exp, MUTED)
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": SURFACE, "edgecolor": AXIS}, medianprops={"color": INK2},
                   whiskerprops={"color": AXIS}, capprops={"color": AXIS})
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=0.8)
        ax.text(i, v.max() + 0.012, f"{v.mean():.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(order)), order, color=INK2)
    ax.set_ylabel("Macro-F1 trung bình 8 aspect", color=INK2)
    ax.set_title("Macro-F1 của từng fold CV (15 fold; số = trung bình)", color=INK, loc="left", fontsize=12)
    save(fig, "cv_folds.png")


def top_words(pipe, k: int) -> dict:
    """Trọng số LinearSVC (one-vs-rest) của các đặc trưng word TF-IDF cho lớp pos và non_pos."""
    model = pipe.named_steps["model"]
    names = model.named_steps["feat"].get_feature_names_out()
    is_word = np.char.startswith(names.astype(str), "word__")
    words = np.array([n[len("word__"):] for n in names])
    out = {}
    for a, est in zip(ASPECTS, model.named_steps["clf"].estimators_):
        classes = list(est.classes_)
        out[a] = {}
        for c in ("pos", "non_pos"):
            coef = np.where(is_word, est.coef_[classes.index(c)], -np.inf)
            idx = np.argsort(coef)[::-1][:k]
            out[a][c] = [(words[i], float(coef[i])) for i in idx]
    return out


def plot_top_words(tw: dict, k: int):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), facecolor=SURFACE)
    for a, ax in zip(ASPECTS, axes.ravel()):
        style_axes(ax, grid_axis="x")
        pos, neg = tw[a]["pos"], tw[a]["non_pos"]
        labels = [w for w, _ in neg][::-1] + [w for w, _ in pos][::-1]
        vals = [-v for _, v in neg][::-1] + [v for _, v in pos][::-1]
        colors = [CLASS_COLOR["non_pos"]] * k + [CLASS_COLOR["pos"]] * k
        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, edgecolor=SURFACE, linewidth=1, height=0.75)
        ax.set_yticks(y, [l.replace("_", " ") for l in labels], fontsize=8.5, color=INK2)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(a, color=INK, fontsize=10.5)
        ax.tick_params(axis="x", labelsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in ("pos", "non_pos")]
    fig.legend(handles, ["đẩy về pos (trọng số > 0)", "đẩy về non_pos (vẽ âm để dễ so)"], frameon=False, ncols=2,
               loc="upper left", bbox_to_anchor=(0.01, 0.955), labelcolor=INK2)
    fig.suptitle(f"Top {k} từ có trọng số lớn nhất của B1-SVM theo aspect (chỉ đặc trưng word, không gồm char n-gram)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "top_words_svm.png")


def plot_class_distribution(train: pd.DataFrame, test: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, sharey=True)
    for ax, (name, df) in zip(axes, [("train", train), ("test", test)]):
        style_axes(ax, grid_axis="x")
        counts = pd.DataFrame({a: df[f"{a}_c3"].value_counts() for a in ASPECTS}).T.reindex(columns=CLASSES3).fillna(0)
        share = counts.div(counts.sum(axis=1), axis=0)
        left = np.zeros(len(ASPECTS))
        y = np.arange(len(ASPECTS))[::-1]
        for c in CLASSES3:
            ax.barh(y, share[c], left=left, color=CLASS_COLOR[c], edgecolor=SURFACE, linewidth=1.5, height=0.7, label=c)
            left += share[c].values
        for yi, n in zip(y, counts["non_pos"]):  # lớp hiếm nhất: ghi số ngay sau thanh, ngoài vùng 100%
            ax.text(1.015, yi, f"{int(n)}", va="center", ha="left", fontsize=8.5, color=CLASS_COLOR["non_pos"],
                    fontweight="bold")
        ax.set_yticks(y, ASPECTS, color=INK2)
        ax.set_xlim(0, 1.09)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{name} ({len(df)} dòng)", color=INK, loc="left", fontsize=11)
    axes[0].legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.07), labelcolor=INK2)
    fig.suptitle("Phân bố lớp theo aspect (số màu cam bên phải = số dòng non_pos)", color=INK, x=0.01, ha="left",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "class_distribution.png")


def plot_timing(cv: pd.DataFrame, order: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, col, title in zip(axes, ["fit_s", "predict_s"], ["Train mỗi fold (giây)", "Dự đoán mỗi fold (giây)"]):
        style_axes(ax, grid_axis="x")
        g = cv.groupby("experiment")[col].mean().reindex(order)
        colors = [MODEL_COLOR["SVM" if "SVM" in e or e == "A1" else "NB"] for e in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, g.values, color=colors, edgecolor=SURFACE, linewidth=1.5, height=0.6)
        for yi, v in zip(y, g.values):
            ax.text(v, yi, f" {v:.2f}s", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y, order, color=INK2)
        ax.set_xlim(0, g.max() * 1.25)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for m in ("SVM", "NB")]
    fig.legend(handles, ["SVM", "Naive Bayes"], frameon=False, ncols=2, loc="upper left", bbox_to_anchor=(0.01, 0.95),
               labelcolor=INK2)
    fig.suptitle("Thời gian trung bình mỗi fold (≈1.800 dòng train, ≈450 dòng dự đoán; chạy tuần tự)", color=INK,
                 x=0.01, ha="left", fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "timing.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=8, help="số từ mỗi lớp trong biểu đồ top từ")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    FIGURES_DIR.mkdir(exist_ok=True)

    train, test = load_split("train"), load_split("test")
    X, Y = train[["comment_norm", "comment_tok"]], train[LABEL_COLS]
    folds = cv_folds(Y.values)

    grid = pd.read_csv(RESULTS_DIR / "cv_results.csv", encoding="utf-8-sig")
    best = grid.loc[grid.groupby("experiment", sort=False)["mean_macro_f1"].idxmax()].set_index("experiment")
    order = ["A1", "A2", "B1-SVM", "B1-NB"]
    cv = run_cv({e: build_best(e, best.loc[e, "params"]) for e in order}, X, Y, folds)
    cv.to_csv(RESULTS_DIR / "figures_cv.csv", index=False, encoding="utf-8-sig")

    base = pd.read_csv(RESULTS_DIR / "baselines_cv.csv", encoding="utf-8-sig")
    base = base.assign(experiment=base["model"], f1=base["mean_macro_f1"])[["experiment", "f1"]]
    scores = pd.concat([base, cv[["experiment", "f1"]]], ignore_index=True)

    plot_metrics(cv, "B1-SVM", "B1-NB")
    plot_cv_folds(scores, ["E0_majority", "E1_rule", *order])
    tw = top_words(joblib.load(MODELS_DIR / "pipeline.pkl"), args.top_k)
    plot_top_words(tw, args.top_k)
    plot_class_distribution(train, test)
    plot_timing(cv, order)

    # Bảng số đi kèm (thay cho đọc số trên hình)
    summary = cv.groupby("experiment", sort=False)[["accuracy", "precision", "recall", "f1", "fit_s", "predict_s"]]
    table = summary.mean().round(3).astype(str) + " ± " + summary.std().round(3).astype(str)
    check = pd.DataFrame({"f1 chạy lại": cv.groupby("experiment", sort=False)["f1"].mean().round(4),
                          "mean_macro_f1 bước 3": best.loc[order, "mean_macro_f1"]})
    lines = ["# Top từ của B1-SVM (word TF-IDF, trọng số LinearSVC)", ""]
    for a in ASPECTS:
        lines += [f"- **{a}** · pos: " + ", ".join(f"{w} ({v:.2f})" for w, v in tw[a]["pos"]),
                  f"  · non_pos: " + ", ".join(f"{w} ({v:.2f})" for w, v in tw[a]["non_pos"])]
    (RESULTS_DIR / "top_words_svm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("## Metric + thời gian (mean ± std, 15 fold)\n\n" + to_markdown(table.astype(object)))
    print("\n## Đối chiếu với bước 3\n\n" + to_markdown(check.astype(object)))
    print("\n" + "\n".join(lines))
    print("\nĐã lưu:", ", ".join(sorted(p.name for p in FIGURES_DIR.glob("*.png"))))


if __name__ == "__main__":
    main()
