# Test report (bước 4)

Model: **B1-SVM** (C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1)), CV trên train 0.7131 ± 0.014. Tập test: 563 dòng, đánh giá một lần (chạy lại với --force: chỉ sửa định dạng bảng báo cáo, model và dự đoán không đổi).

## Kết quả chính

| model | mean_macro_f1 | CI 95% (bootstrap) | mean_accuracy | detection_f1 |
|---|---|---|---|---|
| B1-SVM | 0.721 | 0.689 – 0.745 | 0.914 | 0.889 |
| E1 Rule | 0.708 | 0.680 – 0.728 | 0.887 | 0.884 |
| E0 Majority | 0.290 | 0.287 – 0.292 | 0.771 | 0.497 |

- Chênh lệch B1-SVM − E1 (bootstrap 1000 lần): CI 95% -0.018 – +0.047; B1-SVM cao hơn E1 trong 77% số lần bootstrap.

## Macro-F1 theo aspect

| model | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| B1-SVM | 0.663 | 0.636 | 0.588 | 0.802 | 0.676 | 0.754 | 0.899 | 0.747 |
| E1 Rule | 0.727 | 0.605 | 0.648 | 0.844 | 0.712 | 0.685 | 0.806 | 0.633 |
| E0 Majority | 0.274 | 0.288 | 0.292 | 0.298 | 0.324 | 0.27 | 0.274 | 0.295 |

![F1 theo aspect](../figures/f1_by_aspect.png)

## B1-SVM: precision / recall / F1 từng lớp

| aspect | none | pos | non_pos |
|---|---|---|---|
| sound | P 0.94 · R 0.98 · F1 0.96 (n=394) | P 0.92 · R 0.87 · F1 0.89 (n=156) | P 0.50 · R 0.08 · F1 0.13 (n=13) |
| typing | P 0.91 · R 0.94 · F1 0.93 (n=427) | P 0.76 · R 0.71 · F1 0.73 (n=120) | P 0.38 · R 0.19 · F1 0.25 (n=16) |
| quantity | P 0.44 · R 0.49 · F1 0.46 (n=76) | P 0.88 · R 0.88 · F1 0.88 (n=440) | P 0.49 · R 0.38 · F1 0.43 (n=47) |
| lighting | P 0.97 · R 1.00 · F1 0.98 (n=455) | P 0.85 · R 0.82 · F1 0.83 (n=76) | P 0.73 · R 0.50 · F1 0.59 (n=32) |
| connectivity | P 0.97 · R 1.00 · F1 0.98 (n=531) | P 0.86 · R 0.46 · F1 0.60 (n=13) | P 0.75 · R 0.32 · F1 0.44 (n=19) |
| price | P 0.96 · R 0.99 · F1 0.97 (n=384) | P 0.93 · R 0.90 · F1 0.92 (n=168) | P 0.60 · R 0.27 · F1 0.38 (n=11) |
| delivery | P 0.96 · R 0.99 · F1 0.97 (n=394) | P 0.94 · R 0.90 · F1 0.92 (n=152) | P 0.92 · R 0.71 · F1 0.80 (n=17) |
| service | P 0.94 · R 0.97 · F1 0.96 (n=448) | P 0.83 · R 0.78 · F1 0.81 (n=88) | P 0.58 · R 0.41 · F1 0.48 (n=27) |

## B1-SVM: phát hiện aspect (có nhắc hay không)

| aspect | support (có nhắc) | precision | recall | f1 |
|---|---|---|---|---|
| sound | 169 | 0.96 | 0.852 | 0.903 |
| typing | 136 | 0.8 | 0.706 | 0.75 |
| quantity | 487 | 0.918 | 0.901 | 0.91 |
| lighting | 108 | 0.979 | 0.861 | 0.916 |
| connectivity | 32 | 1.0 | 0.469 | 0.638 |
| price | 179 | 0.97 | 0.905 | 0.936 |
| delivery | 169 | 0.968 | 0.905 | 0.936 |
| service | 115 | 0.873 | 0.774 | 0.82 |
| micro (8 aspect) |  | 0.927 | 0.854 | 0.889 |
| macro (8 aspect) |  | 0.934 | 0.797 | 0.851 |

## B1-SVM: khen hay không khen (chỉ trên các dòng có nhắc theo nhãn đúng)

| aspect | support | macro_f1 (pos/non_pos) | dự đoán none (bỏ sót) |
|---|---|---|---|
| sound | 169 | 0.523 | 25 |
| typing | 136 | 0.541 | 40 |
| quantity | 487 | 0.684 | 48 |
| lighting | 108 | 0.721 | 15 |
| connectivity | 32 | 0.522 | 17 |
| price | 179 | 0.663 | 17 |
| delivery | 169 | 0.869 | 16 |
| service | 115 | 0.692 | 26 |

## B1-SVM: lớp non_pos tách theo điểm gốc (neg 1–4, neu 5–6)

Số dòng được dự đoán đúng lớp / tổng số dòng có điểm gốc thuộc nhóm đó.

| aspect | neg → non_pos | neu → non_pos | pos → pos |
|---|---|---|---|
| sound | 1/9 | 0/4 | 136/156 |
| typing | 1/6 | 2/10 | 85/120 |
| quantity | 7/19 | 11/28 | 386/440 |
| lighting | 5/10 | 11/22 | 62/76 |
| connectivity | 3/10 | 3/9 | 6/13 |
| price | 0/1 | 3/10 | 151/168 |
| delivery | 11/13 | 1/4 | 137/152 |
| service | 8/21 | 3/6 | 69/88 |

## Confusion matrix

![Confusion matrix](../figures/cm_test.png)

