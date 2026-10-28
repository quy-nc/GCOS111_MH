"""Bước 4: đánh giá MỘT LẦN trên test.csv với model đã chọn ở bước 3 (models/pipeline.pkl).

E0 (Majority, lớp nhiều nhất của toàn train) và E1 (Rule-based) được chấm cùng tập test để làm mốc.
Model nhận comment thô (cột comment) để kiểm tra đúng đường dự đoán thật: chuẩn hoá -> tách từ -> TF-IDF -> SVM.

Chốt chặn: sau lần chạy đầu, results/test_report.md tồn tại -> script dừng, trừ khi chạy với --force.
Không chỉnh model sau khi đã xem điểm test.

Input : data/absa/test.csv, data/absa/train.csv (chỉ để lấy lớp majority cho E0), models/pipeline.pkl
Output: results/test_report.md       - metric chính + CI bootstrap, F1 theo aspect, detection, sentiment, neg/neu
        results/test_predictions.csv - nhãn đúng và dự đoán của SVM / E1 cho từng dòng test (dùng ở bước 5)
        figures/f1_by_aspect.png     - Macro-F1 theo aspect: E0, E1, SVM
        figures/cm_test.png          - confusion matrix 3 lớp của SVM cho 8 aspect

Chạy:
    python main.py evaluate
"""

import argparse
import importlib
import json
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from absa_common import (ABSA_DIR, ASPECTS, CLASSES3, LABEL_COLS, PROJECT_DIR, RESULTS_DIR, aspect_scores,  # noqa: E402
                         load_split, to_class4)
from clean_data import to_markdown  # noqa: E402

MODELS_DIR = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "figures"
N_BOOT, BOOT_SEED = 1000, 42

# Màu theo bảng mặc định của skill dataviz (đã chạy validate_palette.js, mode light)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"B1-SVM": "#2a78d6", "E1 Rule": "#eb6834", "E0 Majority": "#1baf7a"}
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fast_macro_f1(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """mean_macro_f1 như aspect_scores nhưng nhanh hơn, dùng cho bootstrap."""
    f1s = []
    for j in range(Y_true.shape[1]):
        yt, yp = Y_true[:, j], Y_pred[:, j]
        per = []
        for c in CLASSES3:
            tp = np.sum((yt == c) & (yp == c))
            denom = np.sum(yt == c) + np.sum(yp == c)
            per.append(0.0 if denom == 0 else 2 * tp / denom)
        f1s.append(np.mean(per))
    return float(np.mean(f1s))


def bootstrap(Y: np.ndarray, preds: dict[str, np.ndarray]) -> tuple[dict, tuple]:
    rng = np.random.default_rng(BOOT_SEED)
    scores = {k: [] for k in preds}
    diff = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(Y), len(Y))
        s = {k: fast_macro_f1(Y[idx], p[idx]) for k, p in preds.items()}
        for k in preds:
            scores[k].append(s[k])
        diff.append(s["B1-SVM"] - s["E1 Rule"])
    ci = {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in scores.items()}
    return ci, (*np.percentile(diff, [2.5, 97.5]), float(np.mean(np.array(diff) > 0)))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_f1_by_aspect(per_aspect: pd.DataFrame, n_test: int, path):
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(ASPECTS))
    w = 0.26
    for i, (name, color) in enumerate(SERIES.items()):
        ax.bar(x + (i - 1) * w, per_aspect.loc[name, ASPECTS], width=w, color=color, label=name,
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x, ASPECTS, color=INK2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 (3 lớp)", color=INK2)
    ax.set_title(f"Macro-F1 theo aspect trên tập test ({n_test} dòng)", color=INK, loc="left", fontsize=12, pad=26)
    # legend nằm ngoài vùng vẽ, ngay dưới tiêu đề, để không đè lên cột
    ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK2,
              borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(Y: np.ndarray, P: np.ndarray, path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), facecolor=SURFACE)
    for j, (a, ax) in enumerate(zip(ASPECTS, axes.ravel())):
        cm = confusion_matrix(Y[:, j], P[:, j], labels=CLASSES3)
        rate = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # tỷ lệ theo hàng = recall từng lớp
        ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{cm[r, c]}\n{rate[r, c]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if rate[r, c] > 0.55 else INK)
        ax.set_xticks(range(3), CLASSES3, fontsize=8, color=INK2)
        ax.set_yticks(range(3), CLASSES3, fontsize=8, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(a, color=INK, fontsize=10)
        if j % 4 == 0:
            ax.set_ylabel("nhãn đúng", color=MUTED, fontsize=8)
        if j >= 4:
            ax.set_xlabel("dự đoán", color=MUTED, fontsize=8)
    fig.suptitle("Confusion matrix của B1-SVM trên test (số dòng và tỷ lệ theo hàng)", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="cho phép chạy lại dù đã đánh giá trên test")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    report_path = RESULTS_DIR / "test_report.md"
    if report_path.exists() and not args.force:
        sys.exit(f"Đã đánh giá trên test ({report_path}). Tập test chỉ dùng một lần; "
                 "nếu buộc phải chạy lại, dùng --force và ghi rõ lý do trong báo cáo.")
    FIGURES_DIR.mkdir(exist_ok=True)

    meta = json.loads((MODELS_DIR / "pipeline_meta.json").read_text(encoding="utf-8"))
    split = json.loads((ABSA_DIR / "split.json").read_text(encoding="utf-8"))
    if meta["input_sha256"] != split["input_sha256"]:
        sys.exit("pipeline.pkl được train trên phiên bản nhãn khác với split hiện tại. Chạy lại bước 1–3.")

    train, test = load_split("train"), load_split("test")
    Y = test[LABEL_COLS].values

    print(f"Dự đoán {len(test)} comment test bằng pipeline.pkl ({meta['experiment']})...", flush=True)
    P_svm = np.asarray(joblib.load(MODELS_DIR / "pipeline.pkl").predict(test["comment"].tolist()))
    rule_predict = importlib.import_module("02_baselines").rule_predict
    P_rule = np.array([rule_predict(t) for t in test["comment_norm"]])
    majority = [train[c].mode()[0] for c in LABEL_COLS]
    P_major = np.tile(majority, (len(test), 1))
    preds = {"B1-SVM": P_svm, "E1 Rule": P_rule, "E0 Majority": P_major}

    scores = {k: aspect_scores(Y, p) for k, p in preds.items()}
    assert abs(fast_macro_f1(Y, P_svm) - scores["B1-SVM"]["mean_macro_f1"]) < 1e-9
    ci, diff_ci = bootstrap(Y, preds)

    summary = pd.DataFrame({
        "mean_macro_f1": [f"{scores[k]['mean_macro_f1']:.3f}" for k in preds],
        "CI 95% (bootstrap)": [f"{ci[k][0]:.3f} – {ci[k][1]:.3f}" for k in preds],
        "mean_accuracy": [f"{scores[k]['mean_accuracy']:.3f}" for k in preds],
        "detection_f1": [f"{scores[k]['detection_f1']:.3f}" for k in preds],
    }, index=pd.Index(list(preds), name="model"))
    per_aspect = pd.DataFrame({k: [scores[k][f"f1_{a}"] for a in ASPECTS] for k in preds}, index=ASPECTS).T.round(3)
    per_aspect.index.name = "model"

    # Báo cáo chi tiết cho model được chọn
    det_rows, sent_rows, cls_rows, nn_rows = [], [], [], []
    scores4 = test[ASPECTS].map(to_class4).values
    for j, a in enumerate(ASPECTS):
        yt, yp = Y[:, j], P_svm[:, j]
        p, r, f, _ = precision_recall_fscore_support(yt != "none", yp != "none", average="binary", zero_division=0)
        det_rows.append({"aspect": a, "support (có nhắc)": int((yt != "none").sum()),
                         "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
        m = yt != "none"
        sent_rows.append({"aspect": a, "support": int(m.sum()),
                          "macro_f1 (pos/non_pos)": round(f1_score(yt[m], yp[m], labels=["pos", "non_pos"],
                                                                     average="macro", zero_division=0), 3),
                          "dự đoán none (bỏ sót)": int((yp[m] == "none").sum())})
        pc, rc, fc, sc = precision_recall_fscore_support(yt, yp, labels=CLASSES3, zero_division=0)
        cls_rows.append({"aspect": a, **{c: f"P {pp:.2f} · R {rr:.2f} · F1 {ff:.2f} (n={s})"
                                         for c, pp, rr, ff, s in zip(CLASSES3, pc, rc, fc, sc)}})
        g4 = scores4[:, j]
        nn = {"aspect": a}
        for c4, target in [("neg", "non_pos"), ("neu", "non_pos"), ("pos", "pos")]:
            mk = g4 == c4
            nn[f"{c4} → {target}"] = f"{int((yp[mk] == target).sum())}/{int(mk.sum())}" if mk.any() else "–"
        nn_rows.append(nn)
    det = pd.DataFrame(det_rows).set_index("aspect")
    p, r, f, _ = precision_recall_fscore_support((Y != "none").ravel(), (P_svm != "none").ravel(), average="binary")
    det.loc["micro (8 aspect)"] = ["", round(p, 3), round(r, 3), round(f, 3)]
    det.loc["macro (8 aspect)"] = ["", *det.iloc[:8][["precision", "recall", "f1"]].astype(float).mean().round(3)]

    out = test[["row_id", "comment", "comment_norm"]].copy()
    for j, a in enumerate(ASPECTS):
        out[f"{a}_true"], out[f"{a}_svm"], out[f"{a}_rule"] = Y[:, j], P_svm[:, j], P_rule[:, j]
    out["n_wrong_svm"] = (P_svm != Y).sum(axis=1)
    out.to_csv(RESULTS_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    plot_f1_by_aspect(per_aspect, len(test), FIGURES_DIR / "f1_by_aspect.png")
    plot_confusion(Y, P_svm, FIGURES_DIR / "cm_test.png")

    lines = [
        "# Test report (bước 4)", "",
        f"Model: **{meta['experiment']}** ({meta['params']}), CV trên train {meta['cv_mean_macro_f1']} ± {meta['cv_std']}. "
        f"Tập test: {len(test)} dòng, đánh giá một lần"
        f"{' (chạy lại với --force: chỉ sửa định dạng bảng báo cáo, model và dự đoán không đổi)' if args.force else ''}.", "",
        "## Kết quả chính", "", to_markdown(summary), "",
        f"- Chênh lệch B1-SVM − E1 (bootstrap {N_BOOT} lần): CI 95% {diff_ci[0]:+.3f} – {diff_ci[1]:+.3f}; "
        f"B1-SVM cao hơn E1 trong {diff_ci[2]:.0%} số lần bootstrap.", "",
        "## Macro-F1 theo aspect", "", to_markdown(per_aspect.astype(object)), "",
        "![F1 theo aspect](../figures/f1_by_aspect.png)", "",
        "## B1-SVM: precision / recall / F1 từng lớp", "", to_markdown(pd.DataFrame(cls_rows).set_index("aspect")), "",
        "## B1-SVM: phát hiện aspect (có nhắc hay không)", "", to_markdown(det.astype(object)), "",
        "## B1-SVM: khen hay không khen (chỉ trên các dòng có nhắc theo nhãn đúng)", "",
        to_markdown(pd.DataFrame(sent_rows).set_index("aspect").astype(object)), "",
        "## B1-SVM: lớp non_pos tách theo điểm gốc (neg 1–4, neu 5–6)", "",
        "Số dòng được dự đoán đúng lớp / tổng số dòng có điểm gốc thuộc nhóm đó.", "",
        to_markdown(pd.DataFrame(nn_rows).set_index("aspect")), "",
        "## Confusion matrix", "", "![Confusion matrix](../figures/cm_test.png)", "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
