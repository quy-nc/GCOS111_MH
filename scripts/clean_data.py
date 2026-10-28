"""Làm sạch dữ liệu review Shopee cho bài toán Aspect-Based Sentiment Analysis (ABSA).

Input : input/ML/merged_data.csv (không bị sửa)
Output: <outdir>/clean_data.csv        - toàn bộ dòng sau khi làm sạch + các cột flag
        <outdir>/clean_data_model.csv  - tập con dùng để gán nhãn / train model
        <outdir>/cleaning_report.md    - thống kê từng bước làm sạch

Chạy:
    python clean_data.py --input ../../../input/ML/merged_data.csv --outdir ../data
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "input" / "ML" / "merged_data.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "data"

# ---------------------------------------------------------------------------
# 1. Sửa product_name (crawler lấy nhầm badge giảm giá "-13%" hoặc "Ad")
# ---------------------------------------------------------------------------
BAD_NAME_RE = re.compile(r"^\s*(-?\d+%|Ad)\s*$")


def name_from_url(url: str) -> str:
    """Lấy tên sản phẩm từ slug URL Shopee: /<ten-san-pham>-i.<shop_id>.<item_id>"""
    path = unquote(urlparse(url).path).lstrip("/")
    path = re.sub(r"-i\.\d+\.\d+$", "", path)
    return re.sub(r"\s+", " ", path.replace("-", " ")).strip()


def categorize(name: str) -> str:
    n = name.lower()
    if "chuột" in n:
        return "chuột"
    if "phím cơ" in n or ("bàn phím" in n and "keycap" not in n) or re.search(r"\baula\b", n):
        return "bàn phím"
    if "keycap" in n:
        return "keycap"
    if re.search(r"\b(tv|tivi|ti vi)\b", n):
        return "tivi"
    if re.search(r"ô tô|xe hơi|\bhud\b", n):
        return "phụ kiện ô tô"
    if re.search(r"điện thoại|oppo|vivo|samsung|iphone|xiaomi redmi|realme|\bzin\b", n):
        return "linh kiện điện thoại"
    if re.search(r"lcd 1602|i2c|arduino", n):
        return "linh kiện điện tử"
    if "màn hình" in n:
        return "màn hình máy tính"
    return "khác"


IN_SCOPE = {"bàn phím", "chuột", "keycap", "màn hình máy tính"}

# ---------------------------------------------------------------------------
# 2. Template của Shopee: "Chất lượng sản phẩm:tốt Tính năng nổi bật:gõ êm ..."
#    Danh sách lấy từ EDA (các nhãn xuất hiện >= ~10 lần).
# ---------------------------------------------------------------------------
TEMPLATE_LABELS = [
    "Chất lượng sản phẩm", "Tính năng nổi bật", "Tính năng tốt nhất", "Đúng với mô tả",
    "Chế độ kết nối", "Thiết kế công thái học", "Độ phân giải", "Độ chính xác",
    "Cách sử dụng", "Độ sắc nét", "Độ êm ái", "Âm thanh", "Độ bền", "Chất lượng",
    "Độ nhạy", "Màu sắc", "Tiếng ồn", "Thiết kế", "Hiệu suất", "Chất liệu", "Độ sáng",
    "Cảm ứng", "Tính năng", "Kết nối", "Ánh sáng", "Màn hình", "Phím cơ", "Cảm giác",
    "Hiển thị", "Hiệu quả", "Độ êm", "Tần số", "Bảo vệ", "Lắp đặt", "Hình ảnh",
    "Giao diện", "Đèn",
]
# Nhãn chung chung, chỉ là câu hỏi gợi ý của Shopee -> bỏ hẳn.
# Các nhãn còn lại (Âm thanh, Độ bền, ...) chính là tên aspect nên giữ lại dạng chữ thường.
GENERIC_LABELS = {"Chất lượng sản phẩm", "Tính năng nổi bật", "Tính năng tốt nhất", "Tính năng"}
# Nhãn dài đặt trước để "Chất lượng sản phẩm" không bị khớp thành "Chất lượng"
TEMPLATE_RE = re.compile(
    r"(?<![\wÀ-ỹ])(" + "|".join(re.escape(l) for l in sorted(TEMPLATE_LABELS, key=len, reverse=True)) + r")\s*:",
)


def _replace_label(m: re.Match) -> str:
    return " , " if m.group(1) in GENERIC_LABELS else f" , {m.group(1)} "


def extract_template(text: str) -> tuple[str, str]:
    """Trả về (text đã xử lý template, chuỗi 'nhãn=giá trị | ...').

    "Chất lượng sản phẩm:tốt Âm thanh:êm  Giao nhanh" -> ", tốt , Âm thanh êm  Giao nhanh"
    """
    matches = list(TEMPLATE_RE.finditer(text))
    if not matches:
        return text, ""
    fields = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Giá trị kết thúc ở nhãn kế tiếp hoặc ở khoảng trắng kép (Shopee nối phần tự do bằng 2 space)
        value = re.split(r"\s{2,}", text[m.end():end].strip(), maxsplit=1)[0].strip()
        fields.append(f"{m.group(1)}={value}")
    # Khoảng trắng kép ngăn cách phần template với phần viết tự do -> thành dấu chấm
    text = re.sub(r"\s{2,}", " . ", TEMPLATE_RE.sub(_replace_label, text))
    return text, " | ".join(fields)


# ---------------------------------------------------------------------------
# 3. Chuẩn hóa văn bản tiếng Việt
# ---------------------------------------------------------------------------
PHONE_RE = re.compile(r"(?:\+?84|0)(?:[\s.-]?\d){8,10}|\d{9,}")  # SĐT, mã đơn hàng
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SDT_LABEL_RE = re.compile(r"\b(sđt|sdt|zalo)\s*:?", re.IGNORECASE)

EMOJI_POS = set("👍❤😍🥰⭐👌💯😁😘🫶🫰🔥🤩✨😊😆🌟✅😎😋💕♥😄😉🙏💖💗💓😻🤗👏🎉")
EMOJI_NEG = set("👎😡😠💔😞😤😒🙁☹😢😩😫🤬🤮😓")
EMOTICON_POS_RE = re.compile(r"[:=;]\)+|[:=;]-\)+|:d+\b|<3+|\(y\)|\^\^+|\^_\^", re.IGNORECASE)
EMOTICON_NEG_RE = re.compile(r"[:=;]\(+|[:=;]-\(+|:'\(+|\(n\)", re.IGNORECASE)

# Kiểu bỏ dấu mới (hoà, thuỷ) -> kiểu cũ (hòa, thủy) để thống nhất từ vựng
_TONE_MOVE = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}
TONE_RE = re.compile(r"(?<![qQ])(" + "|".join(_TONE_MOVE) + r")(?![\wÀ-ỹ])")

# Teencode / viết tắt phổ biến (chỉ thay khi là một từ riêng)
TEENCODE = {
    "k": "không", "ko": "không", "kh": "không", "khum": "không", "hok": "không",
    "hong": "không", "kbt": "không biết", "kb": "không biết",
    "sp": "sản phẩm", "đc": "được", "dc": "được", "mn": "mọi người", "cx": "cũng",
    "vs": "với", "j": "gì", "z": "vậy", "r": "rồi", "mik": "mình", "mk": "mình",
    "siu": "siêu", "zô": "vô", "thik": "thích", "iu": "yêu", "tks": "cảm ơn",
    "thanks": "cảm ơn", "thank": "cảm ơn", "lun": "luôn", "nhìu": "nhiều",
    "oke": "ok", "okela": "ok", "okie": "ok", "oki": "ok", "ukie": "ok",
    "sốp": "shop", "nt": "nhắn tin", "ntn": "như thế nào", "nma": "nhưng mà",
    "ae": "anh em", "đt": "điện thoại", "hàg": "hàng", "qá": "quá", "trc": "trước", "wá": "quá",
    "bth": "bình thường", "bthg": "bình thường", "dth": "dễ thương",
}
TEENCODE_RE = re.compile(r"(?<![\wÀ-ỹ])(" + "|".join(sorted(map(re.escape, TEENCODE), key=len, reverse=True)) + r")(?![\wÀ-ỹ])")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # cũng đưa chữ kiểu 𝐚 về a
    text = URL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)  # bỏ số điện thoại (dữ liệu cá nhân)
    text = SDT_LABEL_RE.sub(" ", text)
    text = "".join(
        " emoji_pos " if ch in EMOJI_POS else " emoji_neg " if ch in EMOJI_NEG else ch
        for ch in text
    )
    text = EMOTICON_POS_RE.sub(" emoji_pos ", text)
    text = EMOTICON_NEG_RE.sub(" emoji_neg ", text)
    # Bỏ các emoji / ký hiệu còn lại (giữ chữ, số, dấu câu cơ bản)
    text = re.sub(r"[^\w\sÀ-ỹ.,!?%/+\-()]", " ", text)
    text = text.lower()
    text = TONE_RE.sub(lambda m: _TONE_MOVE[m.group(1)], text)
    text = re.sub(r"([^\W\d_])\1{2,}", r"\1", text)  # "ngonnnn" -> "ngon"
    text = re.sub(r"([.,!?])\1+", r"\1", text)       # "!!!" -> "!"
    text = TEENCODE_RE.sub(lambda m: TEENCODE[m.group(1)], text)
    text = re.sub(r"(emoji_(?:pos|neg))(\s+\1)+", r"\1", text)  # gộp emoji lặp
    text = re.sub(r"\s*([.,!?])(\s*[.,!?])+", r"\1", text)  # ", ." -> ","
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,.")


TRANSLATED_RE = re.compile(r"^Đã chỉnh sửa|người bán|\bTôi\b")


def to_markdown(table: pd.DataFrame) -> str:
    """Bảng markdown đơn giản (tránh phụ thuộc vào package tabulate)."""
    header = [str(table.index.name or "")] + [str(c) for c in table.columns]
    rows = [[str(i)] + [str(v) for v in r] for i, r in zip(table.index, table.values)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--min-words", type=int, default=4, help="comment ngắn hơn bị flag is_short")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    args.outdir.mkdir(parents=True, exist_ok=True)

    report = []
    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"review_id": "float64"})
    report.append(("Số dòng ban đầu", len(df)))

    df["review_id"] = df["review_id"].astype("int64")
    df["review_time"] = pd.to_datetime(df["review_time"], utc=True)
    df["review_date"] = df["review_time"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date

    bad = df["product_name"].str.match(BAD_NAME_RE)
    df.loc[bad, "product_name"] = df.loc[bad, "product_url"].map(name_from_url)
    df = df.drop(columns=["source_file"])  # tên file bị đặt theo product_name lỗi
    report.append(("product_name sửa lại từ URL", int(bad.sum())))

    df["category"] = df["product_name"].map(categorize)
    df["in_scope"] = df["category"].isin(IN_SCOPE)

    df = df.rename(columns={"comment": "comment_raw"})
    parsed = df["comment_raw"].astype(str).map(extract_template)
    df["template_fields"] = parsed.str[1]
    df["comment_clean"] = parsed.str[0].map(normalize_text)
    report.append(("Comment có template Shopee", int((df["template_fields"] != "").sum())))

    df["n_words"] = df["comment_clean"].str.split().str.len().fillna(0).astype(int)
    df["rating_label"] = pd.cut(df["rating"], [0, 2, 3, 5], labels=["negative", "neutral", "positive"])
    df["maybe_translated"] = df["comment_raw"].str.contains(TRANSLATED_RE)

    empty = df["comment_clean"] == ""
    df = df[~empty]
    report.append(("Bỏ comment rỗng sau làm sạch", int(empty.sum())))

    dup_item = df.duplicated(["item_id", "comment_clean"])
    df = df[~dup_item].copy()
    report.append(("Bỏ trùng trong cùng sản phẩm (item_id + comment_clean)", int(dup_item.sum())))

    df["is_short"] = df["n_words"] < args.min_words
    df["is_dup_text"] = df.duplicated("comment_clean")  # trùng nội dung với sản phẩm khác (giữ bản đầu)

    cols = [
        "review_id", "shop_id", "item_id", "product_name", "category", "in_scope",
        "rating", "rating_label", "review_time", "review_date",
        "comment_raw", "comment_clean", "template_fields", "n_words",
        "is_short", "is_dup_text", "maybe_translated", "product_url", "collection_time_utc",
    ]
    df = df[cols].sort_values(["category", "item_id", "review_time"])
    df.to_csv(args.outdir / "clean_data.csv", index=False, encoding="utf-8-sig")
    report.append(("clean_data.csv", len(df)))

    model = df[df["in_scope"] & ~df["is_short"] & ~df["is_dup_text"]]
    model.to_csv(args.outdir / "clean_data_model.csv", index=False, encoding="utf-8-sig")
    report.append(("clean_data_model.csv (in_scope, >= min_words, không trùng)", len(model)))

    lines = ["# Cleaning report", "", f"Input: `{args.input}`", "", "| Bước | Số dòng |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report]
    lines += ["", "## Flag trên clean_data.csv", "", "| Flag | Số dòng |", "|---|---|"]
    lines += [f"| {c} | {int(df[c].sum())} |" for c in ["in_scope", "is_short", "is_dup_text", "maybe_translated"]]
    for title, frame in [("clean_data.csv", df), ("clean_data_model.csv", model)]:
        lines += ["", f"## Category x rating ({title})", "", to_markdown(pd.crosstab(frame["category"], frame["rating"], margins=True))]
    (args.outdir / "cleaning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
