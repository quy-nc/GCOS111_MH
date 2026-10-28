"""Bước 2: hai baseline để làm mốc so sánh cho SVM / Naive Bayes ở bước 3.

E0 Majority  : mỗi aspect luôn đoán lớp nhiều nhất trong fold-train (không đọc comment).
E1 Rule-based: dò từ khoá aspect + từ khen/chê trong từng mệnh đề của comment_norm (không học từ dữ liệu).

Cả hai được chấm trên đúng các fold CV của bước 3 (repeated 5-fold x 3, chỉ trên train.csv).
test.csv không được đọc ở bước này.

Bộ từ khoá E1 lấy theo quy tắc gán nhãn, chỉnh một lần ngày 2026-10-28 theo lỗi quan sát trên train
("hời", "không có gì chê", "trong tầm", "gửi hàng", "delay" -> typing). Không tinh chỉnh thêm để E1 không
thành "học thuộc" dữ liệu.

Input : data/absa/train.csv
Output: results/baselines_cv.csv     - metric của từng fold
        results/baselines_report.md  - bảng tóm tắt mean ± std, F1 theo aspect, ví dụ E1 đoán sai

Chạy:
    python main.py baselines
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd

from absa_common import ASPECTS, LABEL_COLS, RESULTS_DIR, aspect_scores, cv_folds, load_split
from clean_data import to_markdown

# ---------------------------------------------------------------------------
# E1: từ khoá. Mỗi mục là regex; từ thường được re.escape khi build.
# ---------------------------------------------------------------------------
ASPECT_KW = {
    "sound": ["âm thanh", "tiếng", "tiếng gõ", r"êm(?! tay)", "êm tai", "ồn", "ồn ào", "kêu", "thock", "thocky",
              "creamy", "lạch cạch", "rè", "click", "sướng tai", "đã tai"],
    "typing": ["gõ", "bấm", "phím bấm", "cảm giác", "êm tay", "nhạy", "mượt", "switch", "stab", "stabilizer",
               "kẹt", "nhả phím", "rapid trigger", "lực", "tactile", "linear", "độ nảy",
               "delay"],  # "không delay, bấm nhạy" -> typing; nếu mệnh đề nói về kết nối thì chuyển connectivity (quy tắc 3)
    "quantity": ["chất lượng", "đẹp", "xinh", "cute", "thiết kế", "màu", "ngoại hình", "chắc chắn", "ọp ẹp",
                 "build", "bền", "độ bền", "keycap", "liệt", "double", "lỗi", "hỏng", "như hình", "đúng mô tả",
                 "giống hình", "hoàn thiện", "nhựa", "vỏ", "núm xoay", "layout", "đầm tay", "nặng tay", "chất liệu",
                 "trầy", "xước"],
    "lighting": ["led", "đèn", "rgb", "ánh sáng", "hiệu ứng", "xuyên led"],
    "connectivity": ["kết nối", "bluetooth", r"2\.4g?", "wireless", "không dây", "có dây", "dây", "type c", "type-c",
                     "pin", "sạc", "ngắt", "độ trễ", "receiver", "usb", "mode"],
    "price": ["giá", "giá tiền", "giá cả", "giá thành", "tầm giá", "trong tầm", "tiền", "rẻ", "đắt", "mắc", "hời",
              "đáng tiền", "sale", "đáng mua"],
    "delivery": ["giao", "giao hàng", "gửi hàng", "ship", "shipper", "vận chuyển", "đóng gói", "gói hàng", "bọc",
                 "chống sốc", "hộp", "móp", "nhận hàng", "đến tay"],
    "service": ["shop", "tư vấn", "nhiệt tình", "rep", "hỗ trợ", "bảo hành", "đổi trả", "đổi", "tặng", "quà",
                "sai", "thiếu", "nhân viên", "phản hồi", "chăm sóc", "hướng dẫn"],
}
# Chủ thể chung ("sản phẩm tốt") -> quantity, nhưng chỉ tính khi mệnh đề có từ khen/chê
QUANTITY_WEAK_KW = ["sản phẩm", "bàn phím", "phím cơ", r"(?<!giao )(?<!gửi )(?<!nhận )(?<!gói )hàng"]

POS_WORDS = [
    "tốt", "ổn", "ok", "đẹp", "xinh", "cute", "êm", "êm ái", "êm tai", "nhanh", "nhanh chóng", "nhạy", "mượt",
    "chắc chắn", "cẩn thận", "kỹ", "kĩ", "kỹ càng", "nhiệt tình", "chu đáo", "tận tình", "thân thiện", "rẻ",
    "hợp lý", "đáng tiền", "hời", "ngon", "xịn", "xịn xò", "thích", "ưng", "ưng ý", "hài lòng", "tuyệt", "tuyệt vời",
    "xuất sắc", "đỉnh", "sướng", "sướng tai", "đã tai", "đã tay", "sáng", "trâu", "chuẩn", "đúng mô tả",
    "như hình", "giống hình", "hoàn hảo", "ổn định", "phê", "tiện", "bền", "good", "nice", "best", "perfect",
    "emoji_pos", "khỏi chê", "không có gì chê", "không có gì để chê", "không có gì phải chê", "không chê",
    "hết chỗ chê",
    "miễn chê", "đáng mua", "xứng đáng", "nhiều chế độ", "nhiều hiệu ứng", "creamy", "thock", "thocky",
]
NEG_WORDS = [
    "tệ", "kém", "dở", "dỏm", "lỏ", "cùi", "lỗi", "hỏng", "liệt", "double", "chết", "ồn", "ồn ào", "rè",
    "lạch cạch", "kẹt", "rít", "lag", "delay", "trễ", "ngắt", "mất kết nối", "chập chờn", "chậm", "lâu quá",
    "đắt", "mắc", "sai", "thiếu", "khác mô tả", "không giống", "móp", "méo", "rách", "vỡ", "nát", "ướt",
    "ọp ẹp", "lỏng", "lung lay", "trầy", "xước", "bẩn", "mờ", "bong", "nứt", "buồn", "chán", "thất vọng",
    "tiếc", "mỗi tội", "hơi", "chê", "không rep", "không trả lời", "không phản hồi", "emoji_neg", "bad",
]
NEGATORS = {"không", "chẳng", "chả"}            # đảo nghĩa khi đứng trong 2 từ trước ("không bị lỗi")
UNCERTAIN = {"biết", "chưa"}                     # "chưa biết bền không" -> bỏ qua từ khen/chê đó
RITUAL_RE = re.compile(r"(?<!\w)(cảm ơn shop|cám ơn shop|ủng hộ shop|5 sao|năm sao)(?!\w)")
CLAUSE_SPLIT_RE = re.compile(r"[.,;!?\n]+|(?<!\w)nhưng(?!\w)")


def _alt(patterns: list[str]) -> re.Pattern:
    """Ghép regex theo độ dài giảm dần để cụm dài ("không có gì để chê") khớp trước cụm ngắn ("chê")."""
    parts = [p if any(c in p for c in "\\(?") else re.escape(p) for p in patterns]
    return re.compile(r"(?<!\w)(" + "|".join(sorted(parts, key=len, reverse=True)) + r")(?!\w)")


ASPECT_RE = {a: _alt(kws) for a, kws in ASPECT_KW.items()}
QUANTITY_WEAK_RE = _alt(QUANTITY_WEAK_KW)
POLARITY = {w: "pos" for w in POS_WORDS} | {w: "neg" for w in NEG_WORDS}
POLARITY_RE = _alt(list(POLARITY))


def clause_polarity(clause: str) -> str | None:
    """'neg' nếu mệnh đề có từ chê (sau khi xét phủ định), 'pos' nếu chỉ có từ khen, None nếu không có."""
    found = set()
    for m in POLARITY_RE.finditer(clause):
        word = m.group(1)
        before = clause[:m.start()].split()[-2:]
        if UNCERTAIN & set(before):
            continue
        pol = POLARITY[word]
        if NEGATORS & set(before) and not word.startswith(tuple(NEGATORS)):
            pol = "neg" if pol == "pos" else "pos"
        found.add(pol)
    return "neg" if "neg" in found else "pos" if "pos" in found else None


def rule_predict(text: str) -> list[str]:
    """Trả về 8 nhãn (none / pos / non_pos) theo thứ tự ASPECTS."""
    text = RITUAL_RE.sub(" ", text)
    votes = {a: [] for a in ASPECTS}
    for clause in CLAUSE_SPLIT_RE.split(text):
        if not clause.strip():
            continue
        pol = clause_polarity(clause)
        hit = [a for a in ASPECTS if ASPECT_RE[a].search(clause)]
        if "connectivity" in hit and {m.group(1) for m in ASPECT_RE["typing"].finditer(clause)} == {"delay"}:
            hit.remove("typing")  # "delay" trong mệnh đề nói về kết nối -> chỉ connectivity
        if pol and "quantity" not in hit and QUANTITY_WEAK_RE.search(clause):
            hit.append("quantity")
        if pol and not hit:
            hit = ["quantity"]  # khen/chê chung không gắn aspect cụ thể -> quantity (quy tắc 6)
        for a in hit:
            votes[a].append(pol or "neu")  # nhắc mà không rõ khen chê ~ điểm 6 -> non_pos
    out = []
    for a in ASPECTS:
        v = votes[a]
        out.append("none" if not v else "pos" if "neg" not in v and "pos" in v else "non_pos")
    return out


def summarize(cv: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mean_macro_f1", "mean_accuracy", "detection_f1"]
    g = cv.groupby("model", sort=False)[metrics]
    return pd.DataFrame({m: [f"{a:.3f} ± {s:.3f}" for a, s in zip(g.mean()[m], g.std()[m])] for m in metrics},
                        index=pd.Index(g.mean().index, name="model"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-errors", type=int, default=15, help="số ví dụ E1 đoán sai đưa vào báo cáo")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    RESULTS_DIR.mkdir(exist_ok=True)

    train = load_split("train")
    Y = train[LABEL_COLS].values
    folds = cv_folds(Y)

    # E1 không học từ dữ liệu -> dự đoán một lần cho toàn train, rồi chấm trên từng fold validation
    Y_rule = np.array([rule_predict(t) for t in train["comment_norm"]])

    rows = []
    for k, (tr, va) in enumerate(folds):
        rep, fold = divmod(k, 5)
        majority = [pd.Series(Y[tr, j]).mode()[0] for j in range(len(ASPECTS))]
        Y_major = np.tile(majority, (len(va), 1))
        rows.append({"model": "E0_majority", "repeat": rep, "fold": fold, **aspect_scores(Y[va], Y_major)})
        rows.append({"model": "E1_rule", "repeat": rep, "fold": fold, **aspect_scores(Y[va], Y_rule[va])})
    cv = pd.DataFrame(rows)
    cv.to_csv(RESULTS_DIR / "baselines_cv.csv", index=False, encoding="utf-8-sig")

    per_aspect = cv.groupby("model", sort=False)[[f"f1_{a}" for a in ASPECTS]].mean().round(3)
    per_aspect.columns = ASPECTS

    wrong = np.where((Y_rule != Y).any(axis=1))[0]
    sample = np.random.default_rng(0).choice(wrong, size=min(args.n_errors, len(wrong)), replace=False)

    lines = [
        "# Baselines (bước 2)", "",
        f"Chấm trên {len(folds)} fold validation (repeated 5-fold x 3) của `train.csv` ({len(train)} dòng).", "",
        "## Tóm tắt (mean ± std qua các fold)", "", to_markdown(summarize(cv)), "",
        "- `mean_macro_f1`: metric chính, trung bình Macro-F1 (3 lớp) của 8 aspect.",
        "- `mean_accuracy`: chỉ để tham khảo; E0 có accuracy cao dù không học gì vì lớp `none` áp đảo.",
        "- `detection_f1`: micro-F1 của việc phát hiện aspect có được nhắc hay không.", "",
        "## Macro-F1 theo aspect (mean qua các fold)", "", to_markdown(per_aspect.astype(object)), "",
        f"## {len(sample)} ví dụ E1 đoán sai (trên {len(wrong)}/{len(train)} dòng train có ít nhất 1 aspect sai)", "",
    ]
    for i in sample:
        diff = [f"{a}: đúng `{Y[i, j]}` / E1 `{Y_rule[i, j]}`" for j, a in enumerate(ASPECTS) if Y[i, j] != Y_rule[i, j]]
        lines += [f"- `{train['row_id'][i]}` {train['comment_norm'][i]}", f"  - {' · '.join(diff)}"]
    (RESULTS_DIR / "baselines_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
