"""Bước 6: áp dụng model đã chọn (models/pipeline.pkl) cho comment mới - demo.

Chạy (đường dẫn file tính từ thư mục gốc repo):
    python main.py predict --text "gõ êm, led đẹp nhưng giao hơi chậm"   # một hoặc nhiều --text
    python main.py predict                                               # gõ comment trực tiếp, Enter trống để thoát
    python main.py predict --file data/absa/test.csv --group model       # cả file CSV có cột comment

Với từng comment: lớp của 8 aspect, margin (độ chắc) và các từ đẩy model về lớp đó.
    margin  = điểm decision của lớp được chọn - điểm của lớp đứng thứ hai; gần 0 là model phân vân
    từ      = các từ trong comment có đóng góp lớn nhất (chỉ tính phần word TF-IDF, không tính char n-gram)

Với file:
    results/predict/<tên file>_pred.csv     - file gốc + 8 cột <aspect>_pred
    results/predict/<tên file>_summary.md   - tỉ lệ được nhắc, tỉ lệ khen theo aspect (theo nhóm nếu có --group);
                                              nếu file có sẵn cột <aspect>_c3 thì so với nhãn thật
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from absa_common import ASPECTS, PROJECT_DIR, RESULTS_DIR, prep_frame
from clean_data import extract_template, to_markdown

MODELS_DIR = PROJECT_DIR / "models"
OUT_DIR = RESULTS_DIR / "predict"
VI = {"none": "-", "pos": "khen", "non_pos": "không khen"}
TOP_WORDS = 3


def load_model():
    path = MODELS_DIR / "pipeline.pkl"
    if not path.exists():
        sys.exit(f"Chưa có {path}. Chạy bước 3 trước: python main.py train")
    meta = json.loads((MODELS_DIR / "pipeline_meta.json").read_text(encoding="utf-8"))
    return joblib.load(path), meta


def predict(pipe, texts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Trả về (lớp, margin, polarity, từ ảnh hưởng). Lớp, margin, polarity: DataFrame (n, 8) theo ASPECTS.

    polarity = điểm decision của pos - điểm của non_pos: > 0 nghiêng về khen, < 0 nghiêng về không khen
    (cùng dấu với lớp dự đoán khi aspect được nhắc; vẫn tính cả khi lớp là none).
    """
    model = pipe.named_steps["model"]
    feat, clf = model.named_steps["feat"], model.named_steps["clf"]
    # Comment crawl thô còn template Shopee ("Chất lượng sản phẩm:tốt ..."); file nhãn đã qua bước này ở clean_data.py
    texts = [extract_template(t)[0] for t in texts]
    Xf = feat.transform(prep_frame(texts)).tocsr()
    names = feat.get_feature_names_out()
    # chỉ giải thích bằng từ (bỏ char n-gram và token dấu câu như ",")
    is_word = np.array([n.startswith("word__") and any(ch.isalnum() for ch in n[6:]) for n in names])

    labels, margins, polarity = {}, {}, {}
    reasons = [{} for _ in texts]
    for a, est in zip(ASPECTS, clf.estimators_):
        d = est.decision_function(Xf)
        pred = est.classes_[d.argmax(axis=1)]
        top2 = np.sort(d, axis=1)[:, -2:]
        labels[a], margins[a] = pred, top2[:, 1] - top2[:, 0]
        cls = list(est.classes_)
        polarity[a] = d[:, cls.index("pos")] - d[:, cls.index("non_pos")]
        for i, c in enumerate(pred):
            if c == "none":
                continue
            row = Xf[i]
            contrib = row.data * est.coef_[list(est.classes_).index(c), row.indices]
            order = [k for k in np.argsort(-contrib) if contrib[k] > 0 and is_word[row.indices[k]]]
            reasons[i][a] = [names[row.indices[k]][len("word__"):].replace("_", " ") for k in order[:TOP_WORDS]]
    return pd.DataFrame(labels), pd.DataFrame(margins), pd.DataFrame(polarity), reasons


def show(texts: list[str], pipe) -> None:
    P, M, _, R = predict(pipe, texts)
    norm = prep_frame([extract_template(t)[0] for t in texts])["comment_norm"]
    for i, text in enumerate(texts):
        print(f"\nComment  : {text}\nChuẩn hoá: {norm[i]}")
        print(f"  {'aspect':<13}{'lớp':<12}{'margin':>7}   từ ảnh hưởng")
        for a in ASPECTS:
            c = P.at[i, a]
            if c == "none":
                print(f"  {a:<13}{VI[c]:<12}")
            else:
                print(f"  {a:<13}{VI[c]:<12}{M.at[i, a]:>7.2f}   {', '.join(R[i].get(a, []))}")


def resolve(path: Path) -> Path:
    # main.py chạy script trong scripts/, nên đường dẫn tương đối được tính từ thư mục gốc repo
    return path if path.is_absolute() or path.exists() else PROJECT_DIR / path


def rate_table(L: pd.DataFrame, groups: pd.Series | None) -> pd.DataFrame:
    """Mỗi ô: tỉ lệ khen trong số comment có nhắc aspect, kèm số comment có nhắc."""
    def cell(s: pd.Series) -> str:
        n = int((s != "none").sum())
        return "-" if n == 0 else f"{(s == 'pos').sum() / n:.0%} ({n})"

    parts = [] if groups is None else [L.groupby(groups.values, sort=True).agg(cell)]
    parts.append(L.agg(cell).to_frame("Tất cả").T)
    t = pd.concat(parts)
    t.index.name = groups.name if groups is not None else ""
    return t


def strengths(L: pd.DataFrame, groups: pd.Series | None) -> pd.DataFrame:
    """Aspect được khen nhiều nhất và bị chê (non_pos) nhiều nhất, tính theo số comment."""
    keys = [("Tất cả", L)] if groups is None else [*L.groupby(groups.values, sort=True), ("Tất cả", L)]
    rows = []
    for g, part in keys:
        pos, neg = (part == "pos").sum(), (part == "non_pos").sum()
        rows.append({"nhóm": g, "comment": len(part),
                     "được khen nhiều nhất": f"{pos.idxmax()} ({pos.max()})" if pos.max() else "-",
                     "bị chê nhiều nhất": f"{neg.idxmax()} ({neg.max()})" if neg.max() else "-"})
    return pd.DataFrame(rows).set_index("nhóm")


# Kết luận cho một comment, suy từ lớp của 8 aspect (model không đoán điểm 1-10)
VERDICT = {"pos": "Tích cực", "mixed": "Trái chiều", "non_pos": "Không tích cực", "none": "Không xác định"}
# Kết luận cho cả file theo chỉ số khen = lượt khen / lượt nhắc. Ngưỡng là quy ước của demo, không học từ dữ liệu
OVERALL = [(0.8, "Tích cực"), (0.6, "Khá tích cực, có phàn nàn đáng chú ý"), (0.4, "Trái chiều"), (0.0, "Tiêu cực")]
MIN_MENTIONS = 3  # aspect cần ít nhất ngần này lượt nhắc mới được xếp điểm mạnh / điểm yếu


def comment_verdict(P: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Điểm và kết luận cho từng comment.

    score   = trung bình (điểm khen - điểm không khen) của các aspect được nhắc; NaN nếu không nhắc aspect nào
    verdict = pos (mọi aspect được nhắc đều khen), non_pos (đều không khen), mixed (có cả hai), none (không nhắc)
    """
    n_pos, n_neg = (P == "pos").sum(axis=1), (P == "non_pos").sum(axis=1)
    verdict = np.select([n_pos + n_neg == 0, n_neg == 0, n_pos == 0], ["none", "pos", "non_pos"], "mixed")
    return pd.DataFrame({"score": S.where(P != "none").mean(axis=1), "verdict": verdict}, index=P.index)


def overall_verdict(P: pd.DataFrame) -> dict:
    """Kết luận cả file: chỉ số khen, nhãn kết luận, aspect mạnh nhất / yếu nhất (đủ MIN_MENTIONS lượt nhắc)."""
    pos, men = (P == "pos").sum(), (P != "none").sum()
    share = pos.sum() / men.sum() if men.sum() else float("nan")
    label = next((name for t, name in OVERALL if share >= t), "Không đủ dữ liệu")
    # Xếp theo tỉ lệ khen, hoà thì aspect nhiều lượt nhắc hơn đứng trước (bằng chứng nhiều hơn)
    rate = pd.DataFrame({"rate": pos / men, "men": men})[men >= MIN_MENTIONS]
    best = rate[rate["rate"] == rate["rate"].max()].sort_values("men", ascending=False).head(3)
    worst = rate[rate["rate"] < 1].sort_values(["rate", "men"], ascending=[True, False]).head(2)
    return {"share": share, "n_pos": int(pos.sum()), "n_mentions": int(men.sum()), "label": label,
            "best": best.index.tolist(), "worst": worst.index.tolist(), "rate": pos / men.where(men > 0), "mentions": men}


def run_file(path: Path, col: str, group: str | None, pipe, meta: dict) -> None:
    path = resolve(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if col not in df.columns:
        sys.exit(f"{path.name} không có cột '{col}'. Dùng --col để chọn cột comment.")
    if group and group not in df.columns:
        sys.exit(f"{path.name} không có cột '{group}' để nhóm.")
    texts = df[col].fillna("").astype(str).tolist()
    print(f"Dự đoán {len(texts)} comment trong {path.name} ...", flush=True)
    P, M, S, _ = predict(pipe, texts)
    V = comment_verdict(P, S)
    ov = overall_verdict(P)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for a in ASPECTS:
        out[f"{a}_pred"] = P[a].values
        out[f"{a}_score"] = S[a].where(P[a] != "none").round(2).values
    out["comment_score"] = V["score"].round(2).values
    out["comment_verdict"] = V["verdict"].map(VERDICT).values
    pred_path = OUT_DIR / f"{path.stem}_pred.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    groups = df[group].rename(group) if group else None
    has_truth = all(f"{a}_c3" in df.columns for a in ASPECTS)
    lines = [
        f"# Bước 6 - áp dụng model cho `{path.name}`", "",
        f"- Model: `{meta['experiment']}` ({meta['params']}), train ngày {meta['trained_on']} trên {meta['train_rows']} comment.",
        f"- Số comment: {len(df)}. Nhóm theo: `{group or '(không)'}`.",
        f"- Dự đoán từng dòng: `{pred_path.relative_to(PROJECT_DIR).as_posix()}`.",
        f"- Margin trung bình của các aspect được nhắc: {M.where(P != 'none').stack().mean():.2f} "
        f"(margin < 0.25 ở {(M.where(P != 'none') < 0.25).sum().sum()} ô; gần 0 là model phân vân).", "",
        "## Kết luận tổng", "",
        f"- **{ov['label']}**: chỉ số khen {ov['share']:.0%} ({ov['n_pos']}/{ov['n_mentions']} lượt nhắc aspect là khen). "
        f"Ngưỡng: >= 80% tích cực, 60-79% khá tích cực, 40-59% trái chiều, < 40% tiêu cực (quy ước của demo).",
        "- Kết luận từng comment: " + " · ".join(f"{VERDICT[k]} {int((V['verdict'] == k).sum())}" for k in VERDICT) + ".", "",
        "## Tỉ lệ khen theo aspect (model dự đoán)", "",
        "Mỗi ô: % comment khen trong số comment có nhắc aspect, (số comment có nhắc). "
        "Nhóm có ít comment nhắc tới (ví dụ dưới 10) thì tỉ lệ chưa đáng tin.", "",
        to_markdown(rate_table(P, groups)), "",
        "## Điểm mạnh / điểm yếu (model dự đoán)", "",
        to_markdown(strengths(P, groups)), "",
    ]
    if has_truth:
        T = df[[f"{a}_c3" for a in ASPECTS]].set_axis(ASPECTS, axis=1)
        agree = (P.values == T.values).mean()
        pos_gap = ((P == "pos").sum() - (T == "pos").sum()).astype(int)
        neg_gap = ((P == "non_pos").sum() - (T == "non_pos").sum()).astype(int)
        gap = pd.DataFrame({"khen (dự đoán - thật)": pos_gap, "không khen (dự đoán - thật)": neg_gap}).T
        gap.index.name = ""
        lines += [
            "## So với nhãn thật", "",
            f"File có sẵn nhãn (`<aspect>_c3`), nên so được tóm tắt của model với tóm tắt từ nhãn người gán. "
            f"Tỉ lệ ô (comment × aspect) khớp: {agree:.1%}.", "",
            "### Tỉ lệ khen theo nhãn thật", "",
            to_markdown(rate_table(T, groups)), "",
            "### Chênh lệch số comment (toàn file)", "",
            to_markdown(gap), "",
        ]
    summary_path = OUT_DIR / f"{path.stem}_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã ghi {pred_path.relative_to(PROJECT_DIR).as_posix()} và {summary_path.relative_to(PROJECT_DIR).as_posix()}")
    print("\n" + to_markdown(rate_table(P, groups)))


def main():
    ap = argparse.ArgumentParser(description="Áp dụng model ABSA cho comment mới.")
    ap.add_argument("--text", action="append", help="comment cần dự đoán (lặp lại để nhập nhiều comment)")
    ap.add_argument("--file", type=Path, help="file CSV cần dự đoán")
    ap.add_argument("--col", default="comment", help="tên cột comment trong file (mặc định: comment)")
    ap.add_argument("--group", help="cột để tóm tắt theo nhóm, ví dụ model hoặc shop_name")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    pipe, meta = load_model()
    print(f"Model: {meta['experiment']} ({meta['params']}), CV Macro-F1 {meta['cv_mean_macro_f1']:.3f}")
    if args.file:
        run_file(args.file, args.col, args.group, pipe, meta)
    elif args.text:
        show(args.text, pipe)
    else:
        print("Nhập comment (Enter trống để thoát):")
        while (text := input("> ").strip()):
            show([text], pipe)


if __name__ == "__main__":
    main()
