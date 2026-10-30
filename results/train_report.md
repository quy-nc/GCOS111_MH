# Train report (bước 3)

Repeated 5-fold x 3 trên `train.csv` (2248 dòng), cùng fold với bước 2. Đã thử 80 cấu hình; chi tiết ở `results/cv_results.csv`.

## Cấu hình tốt nhất mỗi thí nghiệm (mean ± std qua 15 fold)

| experiment | params | mean_macro_f1 | mean_accuracy | detection_f1 | fit_time_s |
|---|---|---|---|---|---|
| E0_majority |  | 0.283 ± 0.000 | 0.741 | 0.382 |  |
| E1_rule |  | 0.647 ± 0.011 | 0.854 | 0.846 |  |
| A1 | C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1) | 0.647 ± 0.018 | 0.881 | 0.813 | 0.45 |
| A2 | nb=ComplementNB, alpha=0.1, word.min_df=2, word.ngram_range=(1, 2) | 0.565 ± 0.013 | 0.806 | 0.716 | 0.34 |
| B1-SVM | C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1) | 0.650 ± 0.017 | 0.888 | 0.827 | 3.71 |
| B1-NB | nb=MultinomialNB, alpha=0.01, word.min_df=2, word.ngram_range=(1, 2) | 0.602 ± 0.010 | 0.832 | 0.739 | 1.74 |

## Macro-F1 theo aspect

| experiment | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| E0_majority | 0.281 | 0.272 | 0.242 | 0.299 | 0.324 | 0.276 | 0.274 | 0.292 |
| E1_rule | 0.679 | 0.599 | 0.48 | 0.731 | 0.666 | 0.628 | 0.774 | 0.618 |
| A1 | 0.689 | 0.583 | 0.582 | 0.662 | 0.534 | 0.603 | 0.82 | 0.701 |
| A2 | 0.587 | 0.498 | 0.552 | 0.542 | 0.499 | 0.541 | 0.668 | 0.631 |
| B1-SVM | 0.697 | 0.58 | 0.585 | 0.67 | 0.554 | 0.614 | 0.806 | 0.698 |
| B1-NB | 0.612 | 0.538 | 0.536 | 0.622 | 0.532 | 0.564 | 0.746 | 0.665 |

## Model được chọn

- Cao nhất: **B1-SVM** (0.6504 ± 0.0166); ngưỡng 1 std = 0.6338.
- Thứ tự đơn giản A2 > A1 > B1-NB > B1-SVM → chọn **A1**: C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1).
- Đã fit lại trên toàn bộ train, lưu `models/pipeline.pkl` (nhận comment thô).

## Kiểm tra pipeline.pkl trên 3 comment thô của train

- tốt. ghõ êm. thời gian giao hàng nhanh (2 ngày đã có hàng) phím bấm êm không cs hiện tượng kẹt hơi tiếc là keycap tặng kèm không có xuyên led. cảm quan ban đầu khá ổn
  - sound: `pos` · typing: `pos` · quantity: `pos` · delivery: `pos` · service: `non_pos`
- phím êm hàng tốt nhân viên hỗ trợ mình nhiệt tình sẽ ủng hộ thêm
  - quantity: `pos` · service: `pos`
- hoàn thiện đẹp. màu sắc led rất đẹp. tiếng gõ ok
  - sound: `pos` · typing: `pos` · quantity: `pos` · lighting: `pos`
