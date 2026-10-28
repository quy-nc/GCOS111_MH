"""Bước 3: TF-IDF + SVM / Naive Bayes, chọn model bằng repeated 5-fold CV chỉ trên train.

Thí nghiệm (mỗi aspect một classifier, bọc trong MultiOutputClassifier):
    A1     TF-IDF word + LinearSVC     grid: ngram_range, min_df, C, class_weight
    A2     TF-IDF word + Naive Bayes   grid: ngram_range, min_df, MultinomialNB / ComplementNB, alpha
    B1-SVM TF-IDF word + char_wb 2–5 + LinearSVC     (word TF-IDF lấy theo cấu hình tốt nhất của A1)
    B1-NB  TF-IDF word + char_wb 2–5 + Naive Bayes   (word TF-IDF lấy theo cấu hình tốt nhất của A2)

Metric chọn model: mean_macro_f1 (trung bình Macro-F1 3 lớp của 8 aspect), cùng các fold với bước 2.
Quy tắc chọn: lấy cấu hình có mean cao nhất; nếu thí nghiệm đơn giản hơn kém ít hơn 1 std thì chọn
thí nghiệm đơn giản hơn, thứ tự đơn giản: A2 > A1 > B1-NB > B1-SVM.
test.csv không được đọc ở bước này.

Input : data/absa/train.csv, results/baselines_cv.csv (để so sánh)
Output: results/cv_results.csv      - mọi cấu hình đã thử: tham số, mean ± std, F1 theo aspect, thời gian fit
        results/train_report.md     - cấu hình tốt nhất mỗi thí nghiệm, so với E0/E1, model được chọn
        models/pipeline.pkl         - comment thô -> chuẩn hoá + tách từ -> TF-IDF -> model đã chọn (fit trên toàn train)
        models/pipeline_meta.json   - thí nghiệm, tham số, điểm CV, sha256 file nhãn

Chạy:
    python main.py train
"""

import argparse
import json
import sys
import time
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from absa_common import (ABSA_DIR, ASPECTS, LABEL_COLS, PROJECT_DIR, RESULTS_DIR, aspect_scores, cv_folds,
                         load_split, prep_frame)
from clean_data import to_markdown

MODELS_DIR = PROJECT_DIR / "models"
SIMPLICITY = ["A2", "A1", "B1-NB", "B1-SVM"]  # trái = đơn giản hơn
METRICS = ["mean_macro_f1", "mean_accuracy", "detection_f1", *[f"f1_{a}" for a in ASPECTS]]

TFIDF_GRID = {"feat__word__ngram_range": [(1, 1), (1, 2)], "feat__word__min_df": [1, 2]}
SVM_GRID = {"clf__estimator__C": [0.01, 0.1, 1, 10], "clf__estimator__class_weight": [None, "balanced"]}
NB_GRID = [{"clf__estimator": [nb()], "clf__estimator__alpha": [0.01, 0.1, 0.5, 1.0]}
           for nb in (MultinomialNB, ComplementNB)]


def word_tfidf() -> TfidfVectorizer:
    # comment_tok đã tách từ ("giao_hàng nhanh") -> chỉ cần tách theo khoảng trắng
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, sublinear_tf=True)


def char_tfidf() -> TfidfVectorizer:
    # char n-gram trên comment_norm: chịu được lỗi gõ / không dấu ("ghõ", "leg", "dep")
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def make_pipe(clf, with_char: bool) -> Pipeline:
    parts = [("word", word_tfidf(), "comment_tok")]
    if with_char:
        parts.append(("char", char_tfidf(), "comment_norm"))
    return Pipeline([("feat", ColumnTransformer(parts)), ("clf", MultiOutputClassifier(clf))])


def scorers() -> dict:
    return {m: make_scorer(lambda yt, yp, m=m: aspect_scores(yt, yp)[m]) for m in METRICS}


def run_grid(name: str, pipe: Pipeline, grid, X: pd.DataFrame, Y: pd.DataFrame, folds) -> GridSearchCV:
    print(f"[{name}] {len(ParameterGrid(grid))} cấu hình x {len(folds)} fold ...", flush=True)
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, scoring=scorers(), refit="mean_macro_f1", cv=folds, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(X, Y)
    print(f"[{name}] xong sau {time.time() - t0:.0f}s, best mean_macro_f1 = {gs.best_score_:.4f}", flush=True)
    return gs


def describe(params: dict) -> str:
    out = []
    for k, v in params.items():
        k = k.replace("feat__word__", "word.").replace("clf__estimator__", "").replace("clf__estimator", "nb")
        out.append(f"{k}={type(v).__name__ if k == 'nb' else v}")
    return ", ".join(out)


def results_table(name: str, gs: GridSearchCV) -> pd.DataFrame:
    r = gs.cv_results_
    df = pd.DataFrame({"experiment": name, "params": [describe(p) for p in r["params"]],
                       "fit_time_s": np.round(r["mean_fit_time"], 2)})
    for m in METRICS:
        df[m] = np.round(r[f"mean_test_{m}"], 4)
        df[f"{m}_std"] = np.round(r[f"std_test_{m}"], 4)
    return df.sort_values("mean_macro_f1", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    train = load_split("train")
    X, Y = train[["comment_norm", "comment_tok"]], train[LABEL_COLS]
    folds = cv_folds(Y.values)
    svm = LinearSVC(max_iter=10000)

    searches = {}
    searches["A1"] = run_grid("A1 SVM", make_pipe(svm, False), {**TFIDF_GRID, **SVM_GRID}, X, Y, folds)
    searches["A2"] = run_grid("A2 NB", make_pipe(MultinomialNB(), False), [{**TFIDF_GRID, **g} for g in NB_GRID], X, Y, folds)

    # B1: giữ word TF-IDF tốt nhất của từng model ở trục A, thêm char n-gram, tune lại classifier
    def best_word(gs):
        return {k: [v] for k, v in gs.best_params_.items() if k.startswith("feat__word__")}
    searches["B1-SVM"] = run_grid("B1 SVM", make_pipe(svm, True), {**best_word(searches["A1"]), **SVM_GRID}, X, Y, folds)
    searches["B1-NB"] = run_grid("B1 NB", make_pipe(MultinomialNB(), True),
                                 [{**best_word(searches["A2"]), **g} for g in NB_GRID], X, Y, folds)

    all_cv = pd.concat([results_table(k, gs) for k, gs in searches.items()], ignore_index=True)
    all_cv.to_csv(RESULTS_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")

    # Cấu hình tốt nhất mỗi thí nghiệm + baseline
    best = all_cv.loc[all_cv.groupby("experiment", sort=False)["mean_macro_f1"].idxmax()].set_index("experiment")
    base = pd.read_csv(RESULTS_DIR / "baselines_cv.csv", encoding="utf-8-sig")
    rows = []
    for model, g in base.groupby("model", sort=False):
        rows.append({"experiment": model, "params": "", "mean_macro_f1": f"{g.mean_macro_f1.mean():.3f} ± {g.mean_macro_f1.std():.3f}",
                     "mean_accuracy": f"{g.mean_accuracy.mean():.3f}", "detection_f1": f"{g.detection_f1.mean():.3f}", "fit_time_s": ""})
    for name, r in best.iterrows():
        rows.append({"experiment": name, "params": r["params"],
                     "mean_macro_f1": f"{r.mean_macro_f1:.3f} ± {r.mean_macro_f1_std:.3f}",
                     "mean_accuracy": f"{r.mean_accuracy:.3f}", "detection_f1": f"{r.detection_f1:.3f}",
                     "fit_time_s": r.fit_time_s})
    summary = pd.DataFrame(rows).set_index("experiment")

    per_aspect = pd.concat([
        base.groupby("model", sort=False)[[f"f1_{a}" for a in ASPECTS]].mean(),
        best[[f"f1_{a}" for a in ASPECTS]],
    ]).round(3)
    per_aspect.columns = ASPECTS
    per_aspect.index.name = "experiment"

    # Chọn model: cao nhất, nhưng ưu tiên thí nghiệm đơn giản hơn nếu kém ít hơn 1 std
    top = best["mean_macro_f1"].idxmax()
    threshold = best.loc[top, "mean_macro_f1"] - best.loc[top, "mean_macro_f1_std"]
    chosen = next(e for e in SIMPLICITY if best.loc[e, "mean_macro_f1"] >= threshold)
    gs = searches[chosen]

    final = Pipeline([("prep", FunctionTransformer(prep_frame)), ("model", gs.best_estimator_)])  # đã refit trên toàn train
    joblib.dump(final, MODELS_DIR / "pipeline.pkl")
    split = json.loads((ABSA_DIR / "split.json").read_text(encoding="utf-8"))
    meta = {
        "experiment": chosen, "params": describe(gs.best_params_),
        "cv_mean_macro_f1": round(float(best.loc[chosen, "mean_macro_f1"]), 4),
        "cv_std": round(float(best.loc[chosen, "mean_macro_f1_std"]), 4),
        "train_rows": len(train), "labels": LABEL_COLS,
        "input_sha256": split["input_sha256"], "trained_on": str(date.today()),
    }
    (MODELS_DIR / "pipeline_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Kiểm tra pipeline.pkl chạy được trên comment thô (lấy từ train, không đụng test)
    demo = train["comment"].head(3).tolist()
    demo_pred = joblib.load(MODELS_DIR / "pipeline.pkl").predict(demo)

    lines = [
        "# Train report (bước 3)", "",
        f"Repeated 5-fold x 3 trên `train.csv` ({len(train)} dòng), cùng fold với bước 2. "
        f"Đã thử {len(all_cv)} cấu hình; chi tiết ở `results/cv_results.csv`.", "",
        "## Cấu hình tốt nhất mỗi thí nghiệm (mean ± std qua 15 fold)", "",
        to_markdown(summary.astype(object)), "",
        "## Macro-F1 theo aspect", "", to_markdown(per_aspect.astype(object)), "",
        "## Model được chọn", "",
        f"- Cao nhất: **{top}** ({best.loc[top, 'mean_macro_f1']:.4f} ± {best.loc[top, 'mean_macro_f1_std']:.4f}); "
        f"ngưỡng 1 std = {threshold:.4f}.",
        f"- Thứ tự đơn giản {' > '.join(SIMPLICITY)} → chọn **{chosen}**: {describe(gs.best_params_)}.",
        f"- Đã fit lại trên toàn bộ train, lưu `models/pipeline.pkl` (nhận comment thô).", "",
        "## Kiểm tra pipeline.pkl trên 3 comment thô của train", "",
    ]
    for text, pred in zip(demo, demo_pred):
        lines += [f"- {text}", "  - " + " · ".join(f"{a}: `{p}`" for a, p in zip(ASPECTS, pred) if p != "none")]
    (RESULTS_DIR / "train_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
