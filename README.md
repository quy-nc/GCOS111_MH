# ml-shopee-absa

Aspect-Based Sentiment Analysis (ABSA) cho review bàn phím cơ trên Shopee, dùng **TF-IDF + SVM** và **TF-IDF + Naive Bayes**.

Với mỗi comment, model dự đoán lớp cho từng aspect trong 8 aspect: `sound, typing, quantity, lighting, connectivity, price, delivery, service`.

| Lớp | Điểm gốc | Ý nghĩa |
|---|---|---|
| `none` | -1 | Không nhắc |
| `pos` | 7–10 | Khen |
| `non_pos` | 1–6 | Có nhắc nhưng không khen (gộp neutral và negative) |

> `quantity` là tên cột có sẵn trong file nhãn, hiểu là **chất lượng** (quality).

## Dữ liệu

`data/mechanical_keyboard_labeled_20261028.csv` có 2.811 comment, 15 mẫu bàn phím, 12 shop.

- Cột: `shop_name, model, product_title, comment` cùng 8 cột điểm aspect.
- Điểm aspect là `-1` khi không nhắc, hoặc 1–10 theo mức khen/chê.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy

Chạy từ thư mục gốc của repo bằng `main.py`:

```bash
python main.py --help           # danh sách bước
python main.py prepare          # Bước 1: kiểm tra nhãn, chuẩn hoá + tách từ, chia train/test 80/20
python main.py prepare --seed 7 # tham số phía sau được chuyển nguyên cho script của bước đó
```

Cũng có thể chạy trực tiếp từng script trong `scripts/` (ví dụ `cd scripts && python 01_prepare.py`).

Output của bước 1 nằm trong `data/absa/` (không đưa lên git, chạy lại script để tạo):

| File | Nội dung |
|---|---|
| `train.csv` | Tập train (2.248 dòng): comment đã chuẩn hoá (`comment_norm`) và tách từ (`comment_tok`), 8 cột điểm, 8 cột lớp `<aspect>_c3` |
| `test.csv` | Tập test (563 dòng), cùng cột với `train.csv`. **Chỉ dùng ở bước đánh giá** |
| `split.json` | seed, sha256 của file nhãn, `row_id` của train/test |
| `prepare_report.md` | Các bước lọc, phân bố lớp theo train/test, ví dụ tách từ |

Train/test được chia bằng `MultilabelStratifiedShuffleSplit` (stratify đồng thời theo 8 aspect × 3 lớp), `seed=42`.

## Cấu trúc

```
main.py                             điểm chạy chung: python main.py <bước>
data/                               file nhãn
scripts/
  clean_data.py                     hàm normalize_text (teencode, emoji, ký tự lặp…), dùng lại từ bước làm sạch
  absa_common.py                    danh sách aspect, quy đổi điểm → lớp, vi_tokenize
  01_prepare.py                     Bước 1
  02_baselines.py                   Bước 2
  03_train_cv.py                    Bước 3
  04_evaluate.py                    Bước 4
  04b_figures.py                    Bước 4b
  05_error_analysis.py              Bước 5
  06_predict.py                     Bước 6 (demo dòng lệnh)
  app.py                            Bước 6b (giao diện web Gradio)
results/                            kết quả từng bước (baselines_*, cv_results.csv, train_report.md, test_report.md,
                                    test_predictions.csv, figures_cv.csv, top_words_svm.md, error_analysis.md,
                                    error_samples.csv)
figures/                            f1_by_aspect, cm_test (test) · metrics_svm_vs_nb, cv_folds, top_words_svm,
                                    class_distribution, timing (CV / dữ liệu)
models/                             pipeline.pkl + pipeline_meta.json (không đưa lên git, tạo bằng bước 3)
```

## Kế hoạch

| Bước | Lệnh | Script | Trạng thái |
|---|---|---|---|
| 1. Chuẩn bị dữ liệu, chia train/test | `python main.py prepare` | `01_prepare.py` | ✅ |
| 2. Baseline: Majority, Rule-based (chấm trên CV của train) | `python main.py baselines` | `02_baselines.py` | ✅ |
| 3. TF-IDF + SVM / Naive Bayes, repeated 5-fold CV trên train, lưu `models/pipeline.pkl` | `python main.py train` | `03_train_cv.py` | ✅ |
| 4. Đánh giá một lần trên test (có chốt chặn, chạy lại phải dùng `--force`) | `python main.py evaluate` | `04_evaluate.py` | ✅ |
| 4b. Biểu đồ bổ sung (SVM vs NB, fold CV, top từ, phân bố lớp, thời gian), không đọc lại test | `python main.py figures` | `04b_figures.py` | ✅ |
| 5. Error analysis (đọc lại dự đoán của bước 4, không dự đoán lại) | `python main.py errors` | `05_error_analysis.py` | ✅ |
| 6. Áp dụng model / demo cho comment mới | `python main.py predict` | `06_predict.py` | ✅ |
| 6b. Giao diện web Gradio | `python main.py app` | `app.py` | ✅ |

## Demo (bước 6)

```bash
python main.py predict --text "gõ êm tay, led đẹp, giao hàng nhanh"   # một hoặc nhiều --text
python main.py predict                                                # gõ comment trực tiếp, Enter trống để thoát
python main.py predict --file data/absa/test.csv --group model        # cả file CSV (cần cột comment; --col để đổi tên cột)
```

Với mỗi comment, script in ra lớp của 8 aspect (`khen` / `không khen` / `-`), **margin** (điểm của lớp được chọn trừ điểm của lớp đứng thứ hai; gần 0 nghĩa là model phân vân), và các từ đẩy model về lớp đó (chỉ tính phần word TF-IDF).

Chế độ file ghi `results/predict/<tên file>_pred.csv` và `<tên file>_summary.md`. File summary gồm tỉ lệ khen theo aspect × nhóm và aspect được khen / bị chê nhiều nhất. Nếu file có sẵn cột `<aspect>_c3` thì có thêm phần so với nhãn thật.

### Giao diện web (Gradio)

```bash
python main.py app               # mở http://127.0.0.1:7860
python main.py app --port 7861   # đổi cổng
python main.py app --share       # link công khai tạm thời của Gradio: ai có link đều dùng được, chỉ bật khi cần
```

- **Tab "Một comment"**: nhập comment (hoặc chọn ví dụ), xem chart **điểm khen − không khen** của 8 aspect. Điểm này là decision score của SVM cho lớp `pos` trừ cho lớp `non_pos`, **không phải điểm 1–10**. Bar sang phải (xanh) là nghiêng về khen, sang trái (đỏ) là nghiêng về không khen. Trục cố định ±5 để so được giữa các comment. Rê chuột vào từng dòng để xem margin và từ ảnh hưởng. Aspect có margin < 0.25 được đánh dấu ⚠. Mục "Bảng chi tiết" có cùng số liệu dạng bảng.
- **Tab "File CSV"** (dùng cho file tự crawl): upload CSV (UTF-8, mỗi dòng một review), chọn cột comment (tên cột tuỳ ý) và cột nhóm nếu có. Template Shopee ("Chất lượng sản phẩm:…") được xử lý giống lúc làm sạch dữ liệu train. App bỏ các comment rỗng (review chỉ chấm sao), dự đoán **từng comment** rồi tổng hợp:
  - **Kết luận cho file**: Tích cực / Khá tích cực, có phàn nàn đáng chú ý / Trái chiều / Tiêu cực, theo **chỉ số khen** = lượt khen / lượt nhắc aspect (ngưỡng 80 / 60 / 40%, là quy ước của demo). Kèm số comment, điểm trung bình, phân bố kết luận từng comment, aspect mạnh nhất / cần lưu ý (từ 3 lượt nhắc trở lên), số sao nếu file có cột `rating`, và phần "Cách tính và độ tin cậy";
  - chart số comment khen / không khen theo aspect, kèm % khen (⚠ nếu dưới 10 lượt nhắc);
  - danh sách comment bị đoán "không khen" theo aspect, xếp theo điểm thấp nhất trước;
  - **bảng từng comment**: điểm khen − không khen của 8 aspect, điểm comment (trung bình các aspect được nhắc), kết luận comment (Tích cực / Trái chiều / Không tích cực / Không xác định);
  - bảng theo nhóm và file dự đoán từng dòng để tải về (thêm cột `<aspect>_score`, `comment_score`, `comment_verdict`).
- Không ghép nhiều comment thành một đoạn để dự đoán: thử với 50 comment test, cách ghép cho ra "khen" ở gần như mọi aspect, kể cả aspect có số chê nhiều hơn khen.
- Cache upload của Gradio được ghi vào `.gradio_tmp/` trong repo (không đưa lên git). Analytics của Gradio được tắt.
