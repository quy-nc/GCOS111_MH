"""Điểm chạy chung: gọi từng bước thực nghiệm trong scripts/.

Chạy:
    python main.py prepare                  # Bước 1
    python main.py prepare --seed 7         # tham số phía sau được chuyển nguyên cho script
    python main.py --help                   # danh sách bước
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

# Tên bước -> (script, mô tả). Thêm dòng mới khi có script của bước tiếp theo.
STEPS = {
    "prepare": ("01_prepare.py", "Bước 1: kiểm tra nhãn, chuẩn hoá + tách từ, chia train/test"),
    "baselines": ("02_baselines.py", "Bước 2: baseline E0 (Majority), E1 (Rule-based) trên CV của train"),
    "train": ("03_train_cv.py", "Bước 3: TF-IDF + SVM / Naive Bayes, chọn model bằng CV, lưu models/pipeline.pkl"),
    "evaluate": ("04_evaluate.py", "Bước 4: đánh giá MỘT LẦN trên test (E0, E1, model đã chọn)"),
    "figures": ("04b_figures.py", "Bước 4b: biểu đồ bổ sung (SVM vs NB, fold CV, top từ, phân bố lớp, thời gian)"),
    "errors": ("05_error_analysis.py", "Bước 5: phân tích lỗi của model đã chọn trên test, xuất mẫu câu sai để đọc tay"),
    "predict": ("06_predict.py", "Bước 6: áp dụng model cho comment mới (--text, --file, hoặc gõ trực tiếp)"),
    "app": ("app.py", "Bước 6b: giao diện web Gradio (http://127.0.0.1:7860)"),
}


def main():
    ap = argparse.ArgumentParser(
        description="ABSA bàn phím cơ: chạy từng bước thực nghiệm.",
        epilog="Các bước:\n" + "\n".join(f"  {k:<10} {d}" for k, (_, d) in STEPS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("step", choices=STEPS, help="tên bước cần chạy")
    args, extra = ap.parse_known_args()

    script = SCRIPTS_DIR / STEPS[args.step][0]
    # Chạy trong scripts/ để các script import được absa_common, clean_data
    result = subprocess.run([sys.executable, script.name, *extra], cwd=SCRIPTS_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
