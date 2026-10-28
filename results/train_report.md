# Train report (bước 3)

Repeated 5-fold x 3 trên `train.csv` (2248 dòng), cùng fold với bước 2. Đã thử 80 cấu hình; chi tiết ở `results/cv_results.csv`.

## Cấu hình tốt nhất mỗi thí nghiệm (mean ± std qua 15 fold)

| experiment | params | mean_macro_f1 | mean_accuracy | detection_f1 | fit_time_s |
|---|---|---|---|---|---|
| E0_majority |  | 0.289 ± 0.000 | 0.771 | 0.497 |  |
| E1_rule |  | 0.700 ± 0.009 | 0.879 | 0.881 |  |
| A1 | C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1) | 0.698 ± 0.016 | 0.904 | 0.873 | 0.35 |
| A2 | nb=ComplementNB, alpha=0.1, word.min_df=2, word.ngram_range=(1, 2) | 0.609 ± 0.016 | 0.831 | 0.783 | 0.31 |
| B1-SVM | C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1) | 0.713 ± 0.014 | 0.913 | 0.889 | 3.24 |
| B1-NB | nb=MultinomialNB, alpha=0.01, word.min_df=2, word.ngram_range=(1, 2) | 0.648 ± 0.014 | 0.861 | 0.814 | 1.24 |

## Macro-F1 theo aspect

| experiment | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| E0_majority | 0.274 | 0.288 | 0.292 | 0.298 | 0.323 | 0.27 | 0.274 | 0.295 |
| E1_rule | 0.718 | 0.586 | 0.607 | 0.8 | 0.75 | 0.682 | 0.831 | 0.627 |
| A1 | 0.69 | 0.609 | 0.602 | 0.748 | 0.585 | 0.77 | 0.862 | 0.721 |
| A2 | 0.595 | 0.552 | 0.566 | 0.614 | 0.539 | 0.641 | 0.702 | 0.664 |
| B1-SVM | 0.714 | 0.643 | 0.601 | 0.762 | 0.617 | 0.768 | 0.87 | 0.731 |
| B1-NB | 0.64 | 0.586 | 0.55 | 0.672 | 0.556 | 0.692 | 0.785 | 0.706 |

## Model được chọn

- Cao nhất: **B1-SVM** (0.7131 ± 0.0140); ngưỡng 1 std = 0.6991.
- Thứ tự đơn giản A2 > A1 > B1-NB > B1-SVM → chọn **B1-SVM**: C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1).
- Đã fit lại trên toàn bộ train, lưu `models/pipeline.pkl` (nhận comment thô).

## Kiểm tra pipeline.pkl trên 3 comment thô của train

- tốt. ghõ êm. thời gian giao hàng nhanh (2 ngày đã có hàng) phím bấm êm không cs hiện tượng kẹt hơi tiếc là keycap tặng kèm không có xuyên led. cảm quan ban đầu khá ổn
  - sound: `pos` · typing: `pos` · quantity: `pos` · delivery: `pos` · service: `non_pos`
- phím êm hàng tốt nhân viên hỗ trợ mình nhiệt tình sẽ ủng hộ thêm
  - sound: `pos` · quantity: `pos` · service: `pos`
- chất lượng phù hợp với giá thành. bọc hàng khá kỹ, led đẹp, 11 kiểu led khá ok
  - quantity: `pos` · lighting: `pos` · price: `pos` · delivery: `pos`
