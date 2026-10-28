"""Bước 5: phân tích lỗi của model đã chọn (B1-SVM) trên test. Không sửa model, không dự đoán lại:
chỉ đọc results/test_predictions.csv của bước 4.

Nội dung:
    1. Loại lỗi theo aspect: bỏ sót (nhắc nhưng đoán none), báo nhầm (không nhắc nhưng đoán có), nhầm khen/không khen
    2. Tỷ lệ lỗi theo đặc điểm comment: phủ định, "nhưng/mỗi tội…", không dấu, độ dài, số aspect được nhắc, chữ "êm"
    3. Theo mẫu bàn phím (model): tỷ lệ comment đúng hết 8 aspect, detection F1
    4. SVM và E1 bổ sung cho nhau: số ô (dòng x aspect) chỉ SVM đúng / chỉ E1 đúng
    5. Từ "lạ" trong top trọng số của SVM: xuất hiện trong rất ít comment train -> dấu hiệu học thuộc
    6. Mẫu câu sai để đọc tay (ưu tiên lỗi ở lớp non_pos) -> results/error_samples.csv. Các cột đọc tay
       (nguyen_nhan, ghi_chu, nguoi_danh_gia) được giữ lại theo row_id khi chạy lại script.

Input : results/test_predictions.csv, data/absa/test.csv, data/absa/train.csv, models/pipeline.pkl
Output: results/error_analysis.md, results/error_samples.csv

Chạy:
    python main.py errors
"""

import argparse
import importlib
import re
import sys

import joblib
import numpy as np
import pandas as pd

from absa_common import ASPECTS, PROJECT_DIR, RESULTS_DIR, aspect_scores, load_split
from clean_data import to_markdown

top_words = importlib.import_module("04b_figures").top_words

NEGATION_RE = re.compile(r"(?<!\w)(?:không|chẳng|chả|chưa)(?!\w)")
CONTRAST_RE = re.compile(r"(?<!\w)(?:nhưng|tuy|mỗi tội|tiếc|có điều|chỉ là|chỉ có điều|riêng)(?!\w)")
EM_RE = re.compile(r"(?<!\w)êm(?!\w)")
MANUAL_COLS = ["nguyen_nhan", "ghi_chu", "nguoi_danh_gia"]
CAUSES = {
    "A": "Diễn đạt hiếm / không có từ khoá quen → bỏ sót hoặc gán nhầm aspect",
    "B": "Chê nhẹ nằm trong câu khen (hơi, nhưng, có điều, tiếc…) → đoán pos",
    "C": "Nhãn trung tính (điểm 5–6: nêu tính năng, chấp nhận theo giá) bị gộp vào non_pos",
    "D": "Không dấu / lỗi gõ",
    "E": "Kỳ vọng cá nhân của người mua (quy tắc 42)",
}
VI_CHARS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")


def no_diacritics(text: str) -> bool:
    words = [w for w in re.findall(r"[^\W\d_]+", text) if len(w) > 1]
    return len(words) >= 4 and sum(any(c in VI_CHARS for c in w) for w in words) / len(words) < 0.2


def cue_table(df: pd.DataFrame, wrong_row: np.ndarray, wrong_cell: np.ndarray) -> pd.DataFrame:
    text = df["comment_norm"]
    n_words = text.str.split().str.len()
    n_gold = (df[[f"{a}_true" for a in ASPECTS]] != "none").sum(axis=1)
    cues = {
        "có phủ định (không/chẳng/chưa)": text.str.contains(NEGATION_RE),
        "có đối lập (nhưng/tuy/mỗi tội/tiếc…)": text.str.contains(CONTRAST_RE),
        "không dấu / telex": text.map(no_diacritics),
        "ngắn (≤ 8 từ)": n_words <= 8,
        "trung bình (9–25 từ)": n_words.between(9, 25),
        "dài (> 25 từ)": n_words > 25,
        "nhắc 0–1 aspect": n_gold <= 1,
        "nhắc 2–3 aspect": n_gold.between(2, 3),
        "nhắc ≥ 4 aspect": n_gold >= 4,
    }
    rows = [{"nhóm": "toàn bộ test", "số comment": len(df), "% comment sai ≥1 aspect": wrong_row.mean() * 100,
             "số ô sai / comment": wrong_cell.sum(axis=1).mean()}]
    for name, mask in cues.items():
        m = mask.values
        if m.sum():
            rows.append({"nhóm": name, "số comment": int(m.sum()), "% comment sai ≥1 aspect": wrong_row[m].mean() * 100,
                         "số ô sai / comment": wrong_cell[m].sum(axis=1).mean()})
    out = pd.DataFrame(rows).set_index("nhóm")
    out["% comment sai ≥1 aspect"] = out["% comment sai ≥1 aspect"].round(1)
    out["số ô sai / comment"] = out["số ô sai / comment"].round(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=50, help="số câu sai đưa vào error_samples.csv")
    ap.add_argument("--rare-df", type=int, default=3, help="từ xuất hiện trong <= bao nhiêu comment train thì coi là hiếm")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pred = pd.read_csv(RESULTS_DIR / "test_predictions.csv", encoding="utf-8-sig")
    test, train = load_split("test"), load_split("train")
    pred = pred.merge(test[["row_id", "model"]], on="row_id")
    T = pred[[f"{a}_true" for a in ASPECTS]].values
    S = pred[[f"{a}_svm" for a in ASPECTS]].values
    R = pred[[f"{a}_rule" for a in ASPECTS]].values
    wrong_cell = S != T
    wrong_row = wrong_cell.any(axis=1)

    # 1. Loại lỗi theo aspect
    kinds = pd.DataFrame({
        "ô sai": wrong_cell.sum(axis=0),
        "bỏ sót (nhắc → none)": ((T != "none") & (S == "none")).sum(axis=0),
        "báo nhầm (none → có)": ((T == "none") & (S != "none")).sum(axis=0),
        "pos → non_pos": ((T == "pos") & (S == "non_pos")).sum(axis=0),
        "non_pos → pos": ((T == "non_pos") & (S == "pos")).sum(axis=0),
    }, index=pd.Index(ASPECTS, name="aspect"))
    kinds.loc["tổng"] = kinds.sum()

    # 2. Theo đặc điểm comment
    cues = cue_table(pred, wrong_row, wrong_cell)
    has_em = pred["comment_norm"].str.contains(EM_RE).values
    em_rows = []
    for a in ("sound", "typing"):
        j = ASPECTS.index(a)
        for label, m in (("có 'êm'", has_em), ("không có 'êm'", ~has_em)):
            em_rows.append({"aspect": a, "nhóm": label, "số comment": int(m.sum()),
                            "% sai": round(wrong_cell[m, j].mean() * 100, 1)})
    em = pd.DataFrame(em_rows).set_index("aspect")

    # 3. Theo mẫu bàn phím (chỉ nhóm có ≥ 20 comment test)
    by_model = []
    for model, g in pred.groupby("model"):
        if len(g) < 20:
            continue
        idx = g.index.values
        s = aspect_scores(T[idx], S[idx])
        by_model.append({"model": model, "số comment": len(g), "% đúng cả 8 aspect": round((~wrong_row[idx]).mean() * 100, 1),
                         "detection_f1": round(s["detection_f1"], 3), "mean_macro_f1 (n nhỏ, tham khảo)": round(s["mean_macro_f1"], 3)})
    by_model = pd.DataFrame(by_model).set_index("model").sort_values("% đúng cả 8 aspect")

    # 4. SVM và E1 bổ sung cho nhau
    svm_ok, rule_ok = S == T, R == T
    comp = pd.DataFrame({
        "cả hai đúng": (svm_ok & rule_ok).sum(axis=0),
        "chỉ SVM đúng": (svm_ok & ~rule_ok).sum(axis=0),
        "chỉ E1 đúng": (~svm_ok & rule_ok).sum(axis=0),
        "cả hai sai": (~svm_ok & ~rule_ok).sum(axis=0),
    }, index=pd.Index(ASPECTS, name="aspect"))
    comp.loc["tổng"] = comp.sum()
    oracle = aspect_scores(T, np.where(svm_ok, S, R))["mean_macro_f1"]

    # 5. Từ "lạ" trong top trọng số: ít comment train chứa
    doc_tokens = train["comment_tok"].str.split().map(set)
    tw = top_words(joblib.load(PROJECT_DIR / "models" / "pipeline.pkl"), 15)
    rare = []
    for a in ASPECTS:
        for c in ("pos", "non_pos"):
            for rank, (w, coef) in enumerate(tw[a][c], 1):
                docs = train[doc_tokens.map(lambda s: w in s)]
                if len(docs) <= args.rare_df:
                    labels = docs[f"{a}_c3"].value_counts().to_dict()
                    rare.append({"aspect": a, "lớp": c, "hạng": rank, "từ": w, "trọng số": round(coef, 2),
                                 "số comment train": len(docs), "nhãn aspect trong các comment đó": labels,
                                 "ví dụ": docs["comment_norm"].iloc[0][:100] if len(docs) else ""})
    rare = pd.DataFrame(rare)

    # 6. Mẫu câu sai: ưu tiên dòng có lỗi ở ô mà nhãn đúng là non_pos, rồi dòng sai nhiều ô
    nonpos_err = (wrong_cell & (T == "non_pos")).any(axis=1)
    order = pred.assign(_np=nonpos_err, _nw=wrong_cell.sum(axis=1))[wrong_row]
    order = order.sort_values(["_np", "_nw", "row_id"], ascending=[False, False, True]).head(args.n_samples)
    samples = []
    for i, r in order.iterrows():
        errs = [f"{a}: đúng {T[i, j]} / SVM {S[i, j]} / E1 {R[i, j]}" for j, a in enumerate(ASPECTS) if wrong_cell[i, j]]
        samples.append({"row_id": r["row_id"], "model": r["model"], "comment": r["comment"], "loi": " | ".join(errs)})
    samples = pd.DataFrame(samples)
    samples_path = RESULTS_DIR / "error_samples.csv"
    manual = pd.DataFrame(columns=["row_id", *MANUAL_COLS])
    if samples_path.exists():  # giữ phần đọc tay của lần trước
        old = pd.read_csv(samples_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        manual = old[["row_id", *[c for c in MANUAL_COLS if c in old.columns]]].astype({"row_id": int})
    samples = samples.merge(manual, on="row_id", how="left")
    for c in MANUAL_COLS:
        samples[c] = samples[c].fillna("") if c in samples else ""
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    reviewed = samples["nguyen_nhan"].str.strip() != ""
    cause_count = samples.loc[reviewed, "nguyen_nhan"].str[0].value_counts().reindex(list(CAUSES), fill_value=0)
    cause_table = pd.DataFrame({"nguyên nhân": list(CAUSES.values()), "số comment": cause_count.values},
                               index=pd.Index(list(CAUSES), name="mã"))

    lines = [
        "# Error analysis (bước 5)", "",
        f"Model B1-SVM trên test ({len(pred)} comment). Comment sai ít nhất 1 aspect: {int(wrong_row.sum())} "
        f"({wrong_row.mean():.0%}); tổng số ô (comment × aspect) sai: {int(wrong_cell.sum())}/{wrong_cell.size}.", "",
        "## 1. Loại lỗi theo aspect", "", to_markdown(kinds.astype(object)), "",
        "## 2. Tỷ lệ lỗi theo đặc điểm comment", "",
        "Nhóm không loại trừ nhau (một comment có thể vừa dài vừa có phủ định).", "", to_markdown(cues.astype(object)), "",
        "### Chữ \"êm\" với sound / typing", "", to_markdown(em.reset_index().set_index("aspect").astype(object)), "",
        "## 3. Theo mẫu bàn phím (≥ 20 comment test)", "", to_markdown(by_model.astype(object)), "",
        "## 4. SVM và E1 Rule-based bổ sung cho nhau", "",
        f"Số ô (comment × aspect). Nếu mỗi ô chọn được model đúng (oracle, không làm được trong thực tế) thì "
        f"mean_macro_f1 = {oracle:.3f}, so với SVM {aspect_scores(T, S)['mean_macro_f1']:.3f} và E1 "
        f"{aspect_scores(T, R)['mean_macro_f1']:.3f} → trần trên của việc kết hợp hai cách.", "",
        to_markdown(comp.astype(object)), "",
        f"## 5. Từ trong top 15 trọng số của SVM nhưng chỉ có trong ≤ {args.rare_df} comment train", "",
        f"{len(rare)} từ. Đây là dấu hiệu model học thuộc vài comment cụ thể thay vì quy luật chung.", "",
        to_markdown(rare.set_index("aspect").astype(object)) if len(rare) else "(không có)", "",
        "## 6. Mẫu câu sai để đọc tay", "",
        f"`results/error_samples.csv`: {len(samples)} comment (ưu tiên lỗi ở lớp non_pos). "
        f"Đã đọc tay {int(reviewed.sum())}/{len(samples)} comment (cột `nguyen_nhan`, `ghi_chu`, `nguoi_danh_gia`).", "",
        "Mỗi comment ghi một nguyên nhân chính:", "", to_markdown(cause_table.astype(object)), "",
    ]
    (RESULTS_DIR / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
