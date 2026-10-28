# Baselines (bước 2)

Chấm trên 15 fold validation (repeated 5-fold x 3) của `train.csv` (2248 dòng).

## Tóm tắt (mean ± std qua các fold)

| model | mean_macro_f1 | mean_accuracy | detection_f1 |
|---|---|---|---|
| E0_majority | 0.289 ± 0.000 | 0.771 ± 0.001 | 0.497 ± 0.001 |
| E1_rule | 0.700 ± 0.009 | 0.879 ± 0.005 | 0.881 ± 0.006 |

- `mean_macro_f1`: metric chính, trung bình Macro-F1 (3 lớp) của 8 aspect.
- `mean_accuracy`: chỉ để tham khảo; E0 có accuracy cao dù không học gì vì lớp `none` áp đảo.
- `detection_f1`: micro-F1 của việc phát hiện aspect có được nhắc hay không.

## Macro-F1 theo aspect (mean qua các fold)

| model | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| E0_majority | 0.274 | 0.288 | 0.292 | 0.298 | 0.323 | 0.27 | 0.274 | 0.295 |
| E1_rule | 0.718 | 0.586 | 0.607 | 0.8 | 0.75 | 0.682 | 0.831 | 0.627 |

## 15 ví dụ E1 đoán sai (trên 1387/2248 dòng train có ít nhất 1 aspect sai)

- `706` ngon lắm xiu xiu ngon ăn là mê thích nhất quán này luôn á mmh
  - quantity: đúng `none` / E1 `pos`
- `1783` đèn sáng không chói mắt, ấn nút dễ dàng, đáng đồng tiền bát gạo. mẫu mã đẹp
  - typing: đúng `pos` / E1 `none` · price: đúng `pos` / E1 `non_pos`
- `2579` quá tốt. gõ êm. với giá gần 300k mà có phím red switch. layout 60% hotswap thì quá rẻ rồi 10đ mốt ủng hộ shop tiếp
  - typing: đúng `none` / E1 `pos`
- `1694` đèn led ổn, phím thi thoảng gõ văn bản dài bị ngưng lại, hình in của phím dùng lâu có thể bị mờ nhưng tầm giá này thì không có gì chê
  - typing: đúng `none` / E1 `non_pos`
- `809` bàn phím siêu xinh, tiếng gõ êm tai, hợp lí nhất trong tầm giá nói chung 10000/10 nha. xứng đáng xuống tiền
  - typing: đúng `none` / E1 `pos`
- `2286` mình biết là shop mua hàng từ bên thứ ba nhưng sản phẩm rất ưng nên không có gì phàn nàn cả
  - service: đúng `none` / E1 `non_pos`
- `2375` khong biet. khong biet. tui mua cho ny nên không biết đánh giá gì nữa hihihi
  - price: đúng `none` / E1 `non_pos`
- `125` 100đ. phím cơ red sw. nói thiệt lòng shop rất thành công trong khâu vận chuyển, đặt hôm trước mà hôm sau có liền, đỉnh thiệt chứ về phím thì không có gì để nói nhiều layout 96 phím red sw, êm ái hẳn, không bị tactile, led ok, núm xoay ok. săn sale 360k được 1 con phím này thì không có gì để bàn cãi nữa. phân khúc giá này thì quá hợp lý. mà giao hàng siêu nhanh mặc dù đặt thứ 7 là cn có? trời ơi như 1 cơn gió vậy cha
  - typing: đúng `none` / E1 `non_pos` · service: đúng `none` / E1 `non_pos`
- `1378` độ nhạy ngon gõ êm. thiết kế nhor gonj
  - quantity: đúng `pos` / E1 `non_pos`
- `1391` độ nhạy phím phản hồi nhanh. thiết kế thiết nhỏ gọn, đẹp mắt nha. hài lòng nha troi
  - service: đúng `none` / E1 `pos`
- `2740` tốt. có hotswap rivet. shop giao hàng nhanh, về cơ bản mới nhận thì em này ổn, ưu điểm có hotswap tuy chỉ là rivet nhưng để người mới tập chơi có cái để tập tành mod là ok, 1 mode, cổng type c. stab và switch tất nhiên phải mod lại. nhưng với giá này có một em 98% để vọc vạch tập chơi là ổn. độ bền không đánh giá vì mới dùng và bền hay không tùy người dùng. vote shop 5
  - service: đúng `none` / E1 `pos`
- `230` âm thanh dung quang cao. chất lượng dung quang cao. thiết kế khá ổn, dày và đầm tay. gõ mạnh thì kêu to, gõ nhẹ tay thì kêu nhỏ. không thích ở chỗ là dây cáp liền chứ không phải cổng type-c
  - sound: đúng `pos` / E1 `non_pos` · typing: đúng `none` / E1 `non_pos`
- `53` với tầm giá này thì thực sự phím dùng cực kỳ ổn, chơi game cũng không thấy bị delay
  - typing: đúng `pos` / E1 `non_pos` · quantity: đúng `pos` / E1 `none`
- `498` rất ưng ok ạ shop rep em
  - quantity: đúng `pos` / E1 `none`
- `1825` bàn phím bấm rất êm, không ồn ào, hàng chất lượng, chắc chắn
  - typing: đúng `none` / E1 `pos`
