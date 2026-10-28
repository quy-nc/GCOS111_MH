"""Bước 1: kiểm tra nhãn, chuẩn hoá + tách từ một lần, chia train/test.

Input : data/mechanical_keyboard_labeled_20261028.csv (không bị sửa)
Output: data/absa/train.csv        - tập train: comment, comment_norm, comment_tok, 8 cột điểm, 8 cột lớp 3
        data/absa/test.csv         - tập test, cùng cột với train.csv (chỉ dùng ở bước đánh giá)
        data/absa/split.json       - seed, test_size, sha256 file nhãn, row_id của train/test
        data/absa/prepare_report.md - các bước lọc + phân bố lớp theo train/test

Chạy:
    python 01_prepare.py
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from absa_common import ABSA_DIR, ASPECTS, CLASSES3, DEFAULT_LABELED, normalize_absa, to_class3, vi_tokenize
from clean_data import to_markdown

META_COLS = ["shop_name", "model", "product_title", "comment"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_LABELED)
    ap.add_argument("--outdir", type=Path, default=ABSA_DIR)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.outdir.mkdir(parents=True, exist_ok=True)

    report = []
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df.insert(0, "row_id", range(len(df)))  # vị trí dòng trong file gốc (0-based)
    report.append(("Số dòng ban đầu", len(df)))

    # 1. Kiểm tra cột và điểm
    missing = [c for c in META_COLS + ASPECTS if c not in df.columns]
    if missing:
        sys.exit(f"Thiếu cột: {missing}")
    if df[ASPECTS].isna().any().any():
        sys.exit("Có ô aspect trống:\n" + df[df[ASPECTS].isna().any(axis=1)][["row_id", "comment"]].to_string())
    df[ASPECTS] = df[ASPECTS].astype(int)
    bad = ~df[ASPECTS].isin([-1, *range(1, 11)]).all(axis=1)
    if bad.any():
        sys.exit("Có điểm ngoài -1 / 1–10:\n" + df[bad][["row_id", *ASPECTS]].to_string())

    # 2. Chuẩn hoá + tách từ (chỉ làm một lần, các bước sau dùng comment_tok)
    # Cột comment giữ nguyên bản gốc; bản chuẩn hoá nằm ở comment_norm
    df["comment_norm"] = df["comment"].astype(str).map(normalize_absa)
    changed = df[df["comment_norm"] != df["comment"].astype(str)]
    report.append(("Comment thay đổi khi chuẩn hoá (cột comment giữ nguyên)", len(changed)))
    empty = df["comment_norm"] == ""
    report.append(("Bỏ comment rỗng sau chuẩn hoá", int(empty.sum())))
    df = df[~empty].copy()

    # 3. Bỏ comment trùng sau chuẩn hoá (giữ bản đầu); báo riêng các cặp trùng nhưng nhãn khác nhau
    dup = df.duplicated("comment_norm", keep="first")
    conflict = df[df.duplicated("comment_norm", keep=False)].groupby("comment_norm")[ASPECTS].nunique().gt(1).any(axis=1)
    report.append(("Bỏ comment trùng sau chuẩn hoá", int(dup.sum())))
    report.append(("  trong đó nhóm trùng có nhãn khác nhau", int(conflict.sum())))
    conflict_rows = df[df["comment_norm"].isin(conflict[conflict].index)][["row_id", "comment", *ASPECTS]]
    df = df[~dup].copy()

    print(f"Tách từ {len(df)} comment bằng underthesea...", flush=True)
    df["comment_tok"] = df["comment"].astype(str).map(vi_tokenize)

    # 4. Nhãn 3 lớp
    for a in ASPECTS:
        df[f"{a}_c3"] = df[a].map(to_class3)

    # 5. Chia train/test, stratify đồng thời theo 8 aspect x 3 lớp
    Y = pd.get_dummies(df[[f"{a}_c3" for a in ASPECTS]]).astype(int).values
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_pos, test_pos = next(msss.split(df[["comment_tok"]], Y))
    df["split"] = "train"
    df.iloc[test_pos, df.columns.get_loc("split")] = "test"
    report.append(("Dòng dùng cho thực nghiệm", len(df)))
    report.append(("  train", int((df["split"] == "train").sum())))
    report.append(("  test", int((df["split"] == "test").sum())))

    cols = ["row_id", *META_COLS, "comment_norm", "comment_tok", *ASPECTS, *[f"{a}_c3" for a in ASPECTS]]
    for s in ["train", "test"]:
        df.loc[df["split"] == s, cols].to_csv(args.outdir / f"{s}.csv", index=False, encoding="utf-8-sig")

    split = {
        "input": str(args.input.name),
        "input_sha256": file_sha256(args.input),
        "seed": args.seed,
        "test_size": args.test_size,
        "train_row_ids": df.loc[df["split"] == "train", "row_id"].tolist(),
        "test_row_ids": df.loc[df["split"] == "test", "row_id"].tolist(),
    }
    (args.outdir / "split.json").write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")

    # 6. Báo cáo
    dist = pd.DataFrame({
        (s, c): [int(((df["split"] == s) & (df[f"{a}_c3"] == c)).sum()) for a in ASPECTS]
        for s in ["train", "test"] for c in CLASSES3
    }, index=pd.Index(ASPECTS, name="aspect"))
    dist.columns = [f"{s}/{c}" for s, c in dist.columns]
    for c in CLASSES3:  # tỷ lệ test / tổng để kiểm tra stratify (kỳ vọng ~test_size)
        dist[f"test%/{c}"] = (dist[f"test/{c}"] / (dist[f"train/{c}"] + dist[f"test/{c}"]) * 100).round(1)

    lines = [
        "# Prepare report", "",
        f"Input: `{args.input.name}` · sha256 `{split['input_sha256'][:16]}…` · seed {args.seed} · test_size {args.test_size}", "",
        "| Bước | Số dòng |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in report]
    lines += ["", "## Phân bố lớp 3 theo split", "", to_markdown(dist.astype(object))]  # object: giữ số nguyên không thành 1572.0
    if len(conflict_rows):
        lines += ["", "## Comment trùng nhưng nhãn khác nhau (đã giữ bản đầu)", "",
                  to_markdown(conflict_rows.set_index("row_id"))]
    lines += ["", f"## Ví dụ chuẩn hoá ({min(15, len(changed))}/{len(changed)} comment thay đổi)", ""]
    for _, r in changed.sample(min(15, len(changed)), random_state=args.seed).iterrows():
        lines += [f"- `{r['row_id']}` {r['comment']}", f"  → {r['comment_norm']}"]
    lines += ["", "## 10 ví dụ tách từ", ""]
    for _, r in df.sample(10, random_state=args.seed).iterrows():
        lines += [f"- `{r['row_id']}` {r['comment']}", f"  → {r['comment_tok']}"]
    (args.outdir / "prepare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
