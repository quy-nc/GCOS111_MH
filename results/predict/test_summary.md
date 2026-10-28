# Bước 6 - áp dụng model cho `test.csv`

- Model: `B1-SVM` (C=1, class_weight=balanced, word.min_df=2, word.ngram_range=(1, 1)), train ngày 2026-10-28 trên 2248 comment.
- Số comment: 563. Nhóm theo: `model`.
- Dự đoán từng dòng: `results/predict/test_pred.csv`.
- Margin trung bình của các aspect được nhắc: 1.88 (margin < 0.25 ở 102 ô; gần 0 là model phân vân).

## Kết luận tổng

- **Tích cực**: chỉ số khen 91% (1171/1285 lượt nhắc aspect là khen). Ngưỡng: >= 80% tích cực, 60-79% khá tích cực, 40-59% trái chiều, < 40% tiêu cực (quy ước của demo).
- Kết luận từng comment: Tích cực 446 · Trái chiều 82 · Không tích cực 19 · Không xác định 16.

## Tỉ lệ khen theo aspect (model dự đoán)

Mỗi ô: % comment khen trong số comment có nhắc aspect, (số comment có nhắc). Nhóm có ít comment nhắc tới (ví dụ dưới 10) thì tỉ lệ chưa đáng tin.

| model | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| ATAS ZA981 V4 | 94% (17) | 100% (11) | 94% (50) | 62% (16) | 100% (1) | 100% (24) | 100% (12) | 80% (10) |
| AULA F75 | 100% (29) | 100% (13) | 95% (58) | 94% (18) | 67% (3) | 95% (22) | 92% (13) | 100% (6) |
| AULA F75 HE | 100% (1) | - | 100% (2) | - | - | 100% (2) | - | - |
| AULA F75 MAX | 100% (10) | 100% (1) | 95% (22) | 100% (1) | - | 100% (9) | 50% (6) | 86% (7) |
| AULA S75 PRO | 100% (18) | 75% (4) | 92% (39) | 50% (2) | 67% (3) | 100% (10) | 91% (11) | 94% (16) |
| DAREU EK87 V2 | 100% (3) | 0% (1) | 100% (1) | - | - | 100% (1) | - | - |
| E-DRA EK398S RED SWITCH | 100% (1) | 100% (1) | - | - | - | - | - | - |
| F68 | 100% (15) | 93% (29) | 83% (76) | 78% (9) | 25% (4) | 100% (20) | 91% (23) | 53% (15) |
| K200 | 100% (6) | 100% (17) | 92% (103) | 47% (15) | - | 95% (38) | 96% (57) | 80% (15) |
| K550 V9/K200 | 95% (19) | 100% (12) | 100% (30) | 94% (16) | - | 92% (13) | 100% (11) | 83% (6) |
| MAD68/MAD60 | 100% (8) | 82% (17) | 90% (48) | 60% (5) | 0% (1) | 91% (11) | 83% (12) | 100% (9) |
| MADLIONS MAD60 HE | 100% (8) | 100% (4) | 100% (11) | 100% (3) | 100% (1) | 100% (3) | 100% (1) | 100% (2) |
| Q82 V9 | 100% (6) | 100% (2) | 100% (9) | 100% (4) | - | 100% (3) | 100% (1) | 60% (5) |
| RS01 | 100% (2) | 100% (2) | 100% (4) | 100% (1) | 0% (1) | 100% (1) | 67% (3) | 67% (3) |
| WY - 68K | 100% (7) | 83% (6) | 96% (25) | 80% (5) | 0% (1) | 100% (10) | 88% (8) | 88% (8) |
| Tất cả | 99% (150) | 93% (120) | 92% (478) | 77% (95) | 47% (15) | 97% (167) | 92% (158) | 81% (102) |

## Điểm mạnh / điểm yếu (model dự đoán)

| nhóm | comment | được khen nhiều nhất | bị chê nhiều nhất |
|---|---|---|---|
| ATAS ZA981 V4 | 57 | quantity (47) | lighting (6) |
| AULA F75 | 68 | quantity (55) | quantity (3) |
| AULA F75 HE | 2 | quantity (2) | - |
| AULA F75 MAX | 27 | quantity (21) | delivery (3) |
| AULA S75 PRO | 44 | quantity (36) | quantity (3) |
| DAREU EK87 V2 | 3 | sound (3) | typing (1) |
| E-DRA EK398S RED SWITCH | 1 | sound (1) | - |
| F68 | 91 | quantity (63) | quantity (13) |
| K200 | 113 | quantity (95) | quantity (8) |
| K550 V9/K200 | 38 | quantity (30) | sound (1) |
| MAD68/MAD60 | 56 | quantity (43) | quantity (5) |
| MADLIONS MAD60 HE | 11 | quantity (11) | - |
| Q82 V9 | 10 | quantity (9) | service (2) |
| RS01 | 8 | quantity (4) | connectivity (1) |
| WY - 68K | 34 | quantity (24) | typing (1) |
| Tất cả | 563 | quantity (441) | quantity (37) |

## So với nhãn thật

File có sẵn nhãn (`<aspect>_c3`), nên so được tóm tắt của model với tóm tắt từ nhãn người gán. Tỉ lệ ô (comment × aspect) khớp: 91.4%.

### Tỉ lệ khen theo nhãn thật

| model | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| ATAS ZA981 V4 | 95% (19) | 100% (11) | 88% (50) | 62% (16) | 50% (2) | 96% (26) | 88% (17) | 70% (10) |
| AULA F75 | 94% (32) | 100% (16) | 93% (59) | 75% (20) | 60% (5) | 100% (22) | 93% (15) | 88% (8) |
| AULA F75 HE | 100% (1) | 100% (1) | 100% (1) | - | - | 100% (2) | - | - |
| AULA F75 MAX | 89% (9) | 75% (4) | 85% (20) | 100% (2) | 100% (1) | 100% (10) | 50% (6) | 70% (10) |
| AULA S75 PRO | 100% (19) | 71% (7) | 92% (38) | 100% (3) | 100% (3) | 90% (10) | 92% (12) | 82% (17) |
| DAREU EK87 V2 | 100% (3) | 0% (1) | 100% (1) | - | - | 100% (1) | - | - |
| E-DRA EK398S RED SWITCH | 100% (1) | 100% (1) | - | - | - | - | - | - |
| F68 | 85% (20) | 86% (28) | 82% (83) | 56% (9) | 25% (12) | 87% (23) | 87% (23) | 44% (16) |
| K200 | 88% (8) | 74% (19) | 93% (103) | 50% (20) | 0% (2) | 90% (42) | 95% (59) | 94% (16) |
| K550 V9/K200 | 95% (20) | 100% (13) | 94% (32) | 83% (18) | - | 100% (14) | 100% (13) | 88% (8) |
| MAD68/MAD60 | 83% (12) | 90% (20) | 91% (46) | 60% (5) | 50% (2) | 92% (13) | 85% (13) | 78% (9) |
| MADLIONS MAD60 HE | 100% (9) | 100% (4) | 100% (11) | 100% (3) | 100% (1) | 100% (3) | 100% (1) | 100% (3) |
| Q82 V9 | 100% (7) | 100% (1) | 100% (9) | 100% (6) | - | 100% (3) | 100% (1) | 83% (6) |
| RS01 | 100% (2) | 100% (2) | 100% (7) | 100% (1) | 0% (2) | 100% (1) | 0% (1) | 50% (4) |
| WY - 68K | 71% (7) | 88% (8) | 89% (27) | 60% (5) | 0% (2) | 89% (9) | 88% (8) | 88% (8) |
| Tất cả | 92% (169) | 88% (136) | 90% (487) | 70% (108) | 41% (32) | 94% (179) | 90% (169) | 77% (115) |

### Chênh lệch số comment (toàn file)

|  | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| khen (dự đoán - thật) | -8 | -8 | 1 | -3 | -6 | -6 | -7 | -5 |
| không khen (dự đoán - thật) | -11 | -8 | -10 | -10 | -11 | -6 | -4 | -8 |
