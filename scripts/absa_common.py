"""Hằng số và hàm dùng chung cho thực nghiệm ABSA bàn phím cơ (xem README.md).

Quy đổi điểm theo quy tắc gán nhãn: -1 = không nhắc, 1–4 neg, 5–6 neu, 7–10 pos.
Nhãn chính 3 lớp gộp neu + neg thành non_pos (có nhắc nhưng không khen).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from underthesea import word_tokenize

from clean_data import normalize_text

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ABSA_DIR = DATA_DIR / "absa"
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_LABELED = DATA_DIR / "mechanical_keyboard_labeled_20261028.csv"

ASPECTS = ["sound", "typing", "quantity", "lighting", "connectivity", "price", "delivery", "service"]
LABEL_COLS = [f"{a}_c3" for a in ASPECTS]
CLASSES3 = ["none", "pos", "non_pos"]
CLASSES4 = ["none", "pos", "neu", "neg"]

# CV dùng chung cho bước 2 (baseline) và bước 3 (SVM/NB) để so sánh trên cùng các fold
CV_SPLITS, CV_REPEATS, CV_SEED = 5, 3, 42


def load_split(name: str) -> pd.DataFrame:
    """Đọc data/absa/train.csv hoặc test.csv do 01_prepare.py tạo ra."""
    path = ABSA_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Chưa có {path}. Chạy bước 1 trước: python main.py prepare")
    return pd.read_csv(path, encoding="utf-8-sig")


def cv_folds(Y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated 5-fold x 3, stratify đồng thời theo 8 aspect x 3 lớp. Y: (n, 8) nhãn chuỗi."""
    onehot = pd.get_dummies(pd.DataFrame(Y)).astype(int).values
    cv = RepeatedMultilabelStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=CV_SEED)
    return list(cv.split(np.zeros(len(Y)), onehot))


def aspect_scores(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    """Metric cho một lần dự đoán. Y: (n, 8) nhãn chuỗi theo thứ tự ASPECTS.

    mean_macro_f1 : trung bình Macro-F1 (3 lớp) của 8 aspect -> metric chính để chọn model
    mean_accuracy : trung bình accuracy của 8 aspect (chỉ để tham khảo, dễ ảo vì lớp none áp đảo)
    detection_f1  : micro-F1 của việc phát hiện aspect có được nhắc hay không (none vs khác none)
    f1_<aspect>   : Macro-F1 (3 lớp) của từng aspect
    """
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    f1 = [f1_score(Y_true[:, j], Y_pred[:, j], labels=CLASSES3, average="macro", zero_division=0)
          for j in range(len(ASPECTS))]
    acc = [accuracy_score(Y_true[:, j], Y_pred[:, j]) for j in range(len(ASPECTS))]
    det = f1_score((Y_true != "none").ravel(), (Y_pred != "none").ravel(), zero_division=0)
    return {
        "mean_macro_f1": float(np.mean(f1)),
        "mean_accuracy": float(np.mean(acc)),
        "detection_f1": float(det),
        **{f"f1_{a}": float(v) for a, v in zip(ASPECTS, f1)},
    }


def to_class4(score: int) -> str:
    if score == -1:
        return "none"
    if not 1 <= score <= 10:
        raise ValueError(f"điểm ngoài thang: {score}")
    return "neg" if score <= 4 else "neu" if score <= 6 else "pos"


def to_class3(score: int) -> str:
    c = to_class4(score)
    return "non_pos" if c in ("neg", "neu") else c


# ---------------------------------------------------------------------------
# Chuẩn hoá bổ sung cho normalize_text (clean_data.py không bị sửa)
# ---------------------------------------------------------------------------
# Từ tiếng Anh vốn kết thúc bằng 2 chữ giống nhau -> không gộp (lấy từ dữ liệu ngày 2026-10-28)
EN_DOUBLE_END = {
    "app", "full", "all", "shopee", "shoppee", "wireless", "free", "chill", "tabless", "pass", "off",
    "well", "mall", "bull", "too", "see", "will", "hmm", "scroll", "stress", "overall", "shoo", "dell",
    "miss", "ball",
}
# Từ chỉ gồm chữ cái, không dính chữ số ("10mm", "9xx" được giữ nguyên)
WORD_RE = re.compile(r"(?<!\w)([^\W\d_]+)(?!\w)")

# Teencode chưa có trong TEENCODE của clean_data.py (chỉ thay khi là một từ riêng)
TEENCODE_EXTRA = {
    "nhma": "nhưng mà", "nhg": "nhưng", "mng": "mọi người", "ng": "người", "cs": "có", "nx": "nữa",
    "ns": "nói", "thui": "thôi", "hông": "không", "rùi": "rồi", "gòi": "rồi", "siu": "siêu",
    "oke": "ok", "ôke": "ok",
    "nc": "nói chung", "nhx": "những", "chs": "chơi", "sĩu": "siêu", "bùn": "buồn",
}
TEENCODE_EXTRA_RE = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, TEENCODE_EXTRA), key=len, reverse=True)) + r")(?!\w)")
# "sài" -> "xài", trừ cụm "sơ sài" (chê đóng gói/nhãn mác)
SAI_RE = re.compile(r"(?<!sơ )(?<!\w)sài(?!\w)")


def _collapse_double_end(m: re.Match) -> str:
    """"đẹpp" -> "đẹp", "okee" -> "oke"; normalize_text chỉ gộp chữ lặp từ 3 lần."""
    w = m.group(1)
    if len(w) > 2 and w[-1] == w[-2] and w not in EN_DOUBLE_END:
        return w[:-1]
    return w


def normalize_absa(text: str) -> str:
    """normalize_text + gộp chữ cuối lặp 2 lần + teencode bổ sung."""
    text = normalize_text(text)
    text = WORD_RE.sub(_collapse_double_end, text)
    text = TEENCODE_EXTRA_RE.sub(lambda m: TEENCODE_EXTRA[m.group(1)], text)
    text = SAI_RE.sub("xài", text)
    return re.sub(r"\s+", " ", text).strip()


def vi_tokenize(text: str) -> str:
    """Chuẩn hoá (normalize_absa) rồi tách từ: "giao hàng nhanh" -> "giao_hàng nhanh"."""
    return word_tokenize(normalize_absa(text), format="text")


def prep_frame(texts) -> pd.DataFrame:
    """Comment thô -> DataFrame (comment_norm, comment_tok), giống hệt cách 01_prepare.py tạo 2 cột này.

    Là bước đầu của models/pipeline.pkl để model nhận comment thô khi dự đoán (demo, dữ liệu mới).
    """
    texts = pd.Series(list(texts), dtype=str)
    return pd.DataFrame({"comment_norm": texts.map(normalize_absa), "comment_tok": texts.map(vi_tokenize)})
