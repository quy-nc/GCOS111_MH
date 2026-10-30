# Baselines (bước 2)

Chấm trên 15 fold validation (repeated 5-fold x 3) của `train.csv` (2248 dòng).

## Tóm tắt (mean ± std qua các fold)

| model | mean_macro_f1 | mean_accuracy | detection_f1 |
|---|---|---|---|
| E0_majority | 0.283 ± 0.000 | 0.741 ± 0.001 | 0.382 ± 0.001 |
| E1_rule | 0.647 ± 0.011 | 0.854 ± 0.006 | 0.846 ± 0.007 |

- `mean_macro_f1`: metric chính, trung bình Macro-F1 (3 lớp) của 8 aspect.
- `mean_accuracy`: chỉ để tham khảo; E0 có accuracy cao dù không học gì vì lớp `none` áp đảo.
- `detection_f1`: micro-F1 của việc phát hiện aspect có được nhắc hay không.

## Macro-F1 theo aspect (mean qua các fold)

| model | sound | typing | quantity | lighting | connectivity | price | delivery | service |
|---|---|---|---|---|---|---|---|---|
| E0_majority | 0.281 | 0.272 | 0.242 | 0.299 | 0.324 | 0.276 | 0.274 | 0.292 |
| E1_rule | 0.679 | 0.599 | 0.48 | 0.731 | 0.666 | 0.628 | 0.774 | 0.618 |

## 15 ví dụ E1 đoán sai (trên 1603/2248 dòng train có ít nhất 1 aspect sai)

- `831` êm, ngon trong tầm giá, cảm ơn shop tư vấn nhiệt tình, live xinh quá shop emoji_pos
  - quantity: đúng `none` / E1 `pos`
- `1863` được. có đèn. giao nhanh, chất lượng
  - quantity: đúng `none` / E1 `non_pos` · lighting: đúng `pos` / E1 `non_pos`
- `2574` lúc đầu khong xem kĩ chỉ kết nối dây nên có yêu câu đổi nhưng shoo nhận, thật hên khi mua hub vẫn sử dụng đuocje
  - service: đúng `none` / E1 `pos`
- `1803` tốt. ok nha cho 5s tuy thiếu lót chuột. hàng bao đẹp không
  - service: đúng `pos` / E1 `non_pos`
- `941` âm thanh âm gõ ấm cảm giác bấm rất mượt. độ bền chưa biết. có màn hình hiển thị và có. bàn phím bull chắc không sai lầm khi chọn màu này
  - service: đúng `none` / E1 `pos`
- `2297` tốt. led. sản phẩm phải nói là siêu xịn luôn, led quá đẹp,âm thanh phê cực
  - quantity: đúng `none` / E1 `pos`
- `2386` rapid trigger. hẹ hẹ hẹ lần đầu trải nghiệm he, hy vọng sau này có nhiều tiền hơn sắp con xịn hơn của shop
  - typing: đúng `pos` / E1 `non_pos`
- `133` chất lượng tuyệt vời. bàn phím rất đẹp. cảm giác gõ phím rất đã tay, đặc biệt đang mùa mưa bão mà shop giao tốc hành luôn. cảm ơn shop rất nhiều. nên mua nhé mọi người
  - delivery: đúng `pos` / E1 `non_pos` · service: đúng `none` / E1 `non_pos`
- `1561` giao hàng hơi lâu. bàn phím ok với giá tiền bỏ ra cho shop 5 sao xu
  - quantity: đúng `none` / E1 `pos`
- `1569` hàng đẹp rẻ dùng tốt shop nhiệt tình mỗi tội giao sai màu huhu!
  - quantity: đúng `pos` / E1 `non_pos` · price: đúng `pos` / E1 `non_pos` · delivery: đúng `pos` / E1 `non_pos` · service: đúng `pos` / E1 `non_pos`
- `2737` chất lượng sản phẩm được, giao hàng nhanh, đóng gói sản lhaamr
  - quantity: đúng `none` / E1 `non_pos`
- `256` đã nhận được hàng shop giao nhanh đóng. kỹ lưỡng mong ra thêm nhiều mã nha shop yêu
  - quantity: đúng `none` / E1 `pos` · service: đúng `none` / E1 `pos`
- `62` ván phím khá đẹp nhưng hơi khó nhìn ký tueh phải bật đèn liên tục thì mới nhìn thấy. gõ thấy ok cho shop 5 sao
  - quantity: đúng `non_pos` / E1 `pos` · lighting: đúng `none` / E1 `non_pos` · service: đúng `none` / E1 `pos`
- `531` gõ rất êm, khá chắc chắn nên mua
  - typing: đúng `none` / E1 `pos`
- `1905` ngon. mlem. giá tốt chất lượng ngon anh em nên mua
  - quantity: đúng `none` / E1 `pos` · price: đúng `none` / E1 `pos`
