# Error analysis (bước 5)

Model B1-SVM trên test (563 comment). Comment sai ít nhất 1 aspect: 269 (48%); tổng số ô (comment × aspect) sai: 387/4504.

## 1. Loại lỗi theo aspect

| aspect | ô sai | bỏ sót (nhắc → none) | báo nhầm (none → có) | pos → non_pos | non_pos → pos |
|---|---|---|---|---|---|
| sound | 38 | 25 | 6 | 1 | 6 |
| typing | 72 | 40 | 24 | 3 | 5 |
| quantity | 122 | 48 | 39 | 14 | 21 |
| lighting | 32 | 15 | 2 | 5 | 10 |
| connectivity | 20 | 17 | 0 | 2 | 1 |
| price | 30 | 17 | 5 | 1 | 7 |
| delivery | 25 | 16 | 5 | 1 | 3 |
| service | 48 | 26 | 13 | 3 | 6 |
| tổng | 387 | 204 | 94 | 30 | 59 |

## 2. Tỷ lệ lỗi theo đặc điểm comment

Nhóm không loại trừ nhau (một comment có thể vừa dài vừa có phủ định).

| nhóm | số comment | % comment sai ≥1 aspect | số ô sai / comment |
|---|---|---|---|
| toàn bộ test | 563 | 47.8 | 0.69 |
| có phủ định (không/chẳng/chưa) | 167 | 62.3 | 0.99 |
| có đối lập (nhưng/tuy/mỗi tội/tiếc…) | 80 | 73.8 | 1.18 |
| không dấu / telex | 8 | 87.5 | 1.12 |
| ngắn (≤ 8 từ) | 55 | 18.2 | 0.25 |
| trung bình (9–25 từ) | 372 | 45.7 | 0.61 |
| dài (> 25 từ) | 136 | 65.4 | 1.07 |
| nhắc 0–1 aspect | 88 | 39.8 | 0.5 |
| nhắc 2–3 aspect | 387 | 43.4 | 0.59 |
| nhắc ≥ 4 aspect | 88 | 75.0 | 1.31 |

### Chữ "êm" với sound / typing

| aspect | nhóm | số comment | % sai |
|---|---|---|---|
| sound | có 'êm' | 98 | 8.2 |
| sound | không có 'êm' | 465 | 6.5 |
| typing | có 'êm' | 98 | 15.3 |
| typing | không có 'êm' | 465 | 12.3 |

## 3. Theo mẫu bàn phím (≥ 20 comment test)

| model | số comment | % đúng cả 8 aspect | detection_f1 | mean_macro_f1 (n nhỏ, tham khảo) |
|---|---|---|---|---|
| K550 V9/K200 | 38 | 42.1 | 0.88 | 0.583 |
| AULA S75 PRO | 44 | 43.2 | 0.915 | 0.634 |
| ATAS ZA981 V4 | 57 | 43.9 | 0.877 | 0.669 |
| WY - 68K | 34 | 50.0 | 0.861 | 0.66 |
| F68 | 91 | 50.5 | 0.894 | 0.721 |
| MAD68/MAD60 | 56 | 51.8 | 0.883 | 0.698 |
| AULA F75 | 68 | 54.4 | 0.885 | 0.674 |
| AULA F75 MAX | 27 | 55.6 | 0.898 | 0.615 |
| K200 | 113 | 61.1 | 0.892 | 0.649 |

## 4. SVM và E1 Rule-based bổ sung cho nhau

Số ô (comment × aspect). Nếu mỗi ô chọn được model đúng (oracle, không làm được trong thực tế) thì mean_macro_f1 = 0.857, so với SVM 0.721 và E1 0.708 → trần trên của việc kết hợp hai cách.

| aspect | cả hai đúng | chỉ SVM đúng | chỉ E1 đúng | cả hai sai |
|---|---|---|---|---|
| sound | 492 | 33 | 22 | 16 |
| typing | 419 | 72 | 34 | 38 |
| quantity | 381 | 60 | 74 | 48 |
| lighting | 520 | 11 | 19 | 13 |
| connectivity | 532 | 11 | 9 | 11 |
| price | 494 | 39 | 12 | 18 |
| delivery | 519 | 19 | 10 | 15 |
| service | 435 | 80 | 23 | 25 |
| tổng | 3792 | 325 | 203 | 184 |

## 5. Từ trong top 15 trọng số của SVM nhưng chỉ có trong ≤ 3 comment train

32 từ. Đây là dấu hiệu model học thuộc vài comment cụ thể thay vì quy luật chung.

| aspect | lớp | hạng | từ | trọng số | số comment train | nhãn aspect trong các comment đó | ví dụ |
|---|---|---|---|---|---|---|---|
| sound | pos | 12 | chất_âm | 1.0 | 3 | {'pos': 2, 'non_pos': 1} | bàn phím giao nhanh chất âm tốt có đầy đủ hướng dẫn sử dụng 826 |
| sound | non_pos | 6 | gõ_phím | 0.97 | 2 | {'pos': 1, 'non_pos': 1} | quá ngon trong tầm giá. 1 con phím cực kì ổn trong tầm giá, build mình thấy cũng khá chắc chắn, âm t |
| sound | non_pos | 7 | câu | 0.92 | 2 | {'non_pos': 1, 'none': 1} | độ bền chắc tay. ánh sáng tốt. phím cơ ồn vl nhưng đầm tay. sản phẩm tốt nhưng bên hỗ trợ thường hay |
| typing | non_pos | 12 | lâu_lâu | 0.97 | 3 | {'non_pos': 2, 'none': 1} | âm thanh đúng gu mình, màu sắc đẹp, newbie dùng tốt, shop tư vấn chu đáo, đi kèm hướng dẫn, tận tâm, |
| quantity | non_pos | 6 | ăn | 1.24 | 3 | {'non_pos': 3} | hàng như hình. mỗi tội có 4 phím không ăn. mình phải mua riêng swift thay lại. mẫu mã đẹp. nhựa xịn  |
| quantity | non_pos | 7 | đôi_khi | 1.2 | 3 | {'non_pos': 2, 'pos': 1} | âm nghe vui tai. 3 mode kết nối, build đầm. shop giao hàng siêu tốc luôn ạ. mình đặt địa chỉ ở quê n |
| quantity | non_pos | 9 | bẩn | 1.16 | 3 | {'non_pos': 2, 'pos': 1} | ok. không có. bàn phím ok, dùng lâu sẽ dễ bám bụi bẩn. gõ êm, với tầm giá ok khi mua với con phim nà |
| quantity | non_pos | 13 | xước | 1.03 | 3 | {'non_pos': 2, 'pos': 1} | gõ rất sướng, màu đẹp, sản phẩm hoàn thiện không vết xước |
| quantity | non_pos | 15 | chông | 1.01 | 2 | {'non_pos': 1, 'pos': 1} | sản phẩm khá ok bàn phím hơi chông chênh xíu ship hàng khá nhanh |
| lighting | pos | 5 | led_ok | 1.37 | 3 | {'pos': 3} | giống như mô tả hình ảnh, bàn phím đẹp,led ok, không bị lỗi hay bất cứ gì hêts |
| lighting | pos | 8 | led_ổn | 1.29 | 2 | {'pos': 2} | độ nhạy tốt. thiết kế đẹp led ổn. chất liệu nhựa tốt. kt phím không bị lỗi còn kết đủ loại thiết bị |
| lighting | pos | 12 | nhạc | 0.91 | 2 | {'pos': 2} | ngon như giá. leg theo nhạc emoji_pos. ngon |
| lighting | non_pos | 6 | nhảy | 1.09 | 3 | {'non_pos': 2, 'none': 1} | hàng đẹp giá tốt xài êm đầy đủ phụ kiện, 3 mode connect - 3 channel bt, màn hình tùy chỉnh gif 140fp |
| lighting | non_pos | 10 | phím_số | 0.95 | 3 | {'non_pos': 2, 'none': 1} | bàn phím số, nút vặn điều chỉnh âm thanh và chế độ cũng như cường độ đèn bàn phím. tốt. khuyến khích |
| lighting | non_pos | 13 | led_phím | 0.93 | 3 | {'none': 1, 'non_pos': 1, 'pos': 1} | thấy ok, độ nhạy phím 90%, trải nghiệm 20 ngày nay chưa phải sạc lần nào, mình làm văn phòng nên khô |
| connectivity | pos | 2 | kết | 1.09 | 3 | {'pos': 2, 'none': 1} | best. led tuyệt cà là vời. sản phẩm gọn, nặng tay, dày. shop rất tận tâm trong việc hướng dẫn và giả |
| connectivity | pos | 3 | cấm | 0.93 | 2 | {'pos': 1, 'non_pos': 1} | tốt. ok. shop đóng gói nhanh, hàng đúng với mô tả, sản phẩm mới cấm vào laptop nhận được hoạt động t |
| connectivity | pos | 10 | hạt | 0.83 | 2 | {'pos': 1, 'none': 1} | bifnhb thuong. cầm chắc tay. giá hạt rẻ dây có thể đi gọn, cầm phím không lọc xọc mong là bền |
| connectivity | pos | 12 | bền_pin | 0.81 | 2 | {'pos': 1, 'non_pos': 1} | ok. quá ưng quá đẹp quá bền pin cũng ok đầy đủ phụ kiện âm trầm ấm êm tai |
| connectivity | non_pos | 7 | cùi | 0.97 | 2 | {'non_pos': 2} | bàn phím thì đẹp nhưng pin cùi quá dùng mà mở led full chắc được tầm 5 6 tiếng, thấy pin tụt nhanh l |
| connectivity | non_pos | 8 | wireless | 0.95 | 3 | {'non_pos': 3} | tốt, key không có xuyên led. rgb led, lcd screen, wireless, về thay keycap xuyên led là đầy đủ tính  |
| connectivity | non_pos | 11 | dây_cáp | 0.87 | 2 | {'non_pos': 2} | cũng tốt. có led nhiều màu, núm xoay tăng giảm âm lượng và độ sáng đèn bàn phím. h mới có thể đánh g |
| connectivity | non_pos | 13 | việt | 0.78 | 2 | {'none': 1, 'non_pos': 1} | bàn phím số, nút vặn điều chỉnh âm thanh và chế độ cũng như cường độ đèn bàn phím. tốt. khuyến khích |
| price | non_pos | 6 | sản_phẩn | 0.85 | 2 | {'none': 1, 'non_pos': 1} | chất lượng sản phẩn xài ổn nhưng có 1 switch bị lỗi phải tự thay bằng switch dự phòng nên mình giảm  |
| price | non_pos | 13 | tầm_trung | 0.72 | 2 | {'non_pos': 1, 'none': 1} | phím ở mức giá tầm trung nên tiếng gõ cũng ok, nhưng thích ồn hơn nữa. mọi người cứ tham khảo rồi mu |
| delivery | pos | 13 | dong | 0.75 | 2 | {'pos': 2} | âm thanh ngon, creamy. màn hình on ap, chất lượng build ngon, không bi loc xoc. emoji_pos, dong goi  |
| delivery | non_pos | 9 | lâu_tí | 0.93 | 2 | {'non_pos': 2} | độ bền ok nha. chất lượng chất lượng ok, gõ êm. bấm khá nhạy nha. vì đang chuẩn bị tết nên hàng giao |
| delivery | non_pos | 11 | khâu | 0.88 | 3 | {'pos': 2, 'non_pos': 1} | 100đ. phím cơ red sw. nói thiệt lòng shop rất thành công trong khâu vận chuyển, đặt hôm trước mà hôm |
| delivery | non_pos | 15 | rớt | 0.79 | 3 | {'non_pos': 2, 'none': 1} | sản phẩm ok giao hơi nhanh chứ không nhanh lắm cũng đẹp mà mới tháo ra nút enter bị rớt nhưng gắn lạ |
| service | pos | 13 | chuan | 1.17 | 2 | {'none': 1, 'pos': 1} | âm thanh chuan, ro, to. màn hình dep, monh. chất lượng cung cap, chắc chắn. chuẩn, đẹp, cứng cáp, ch |
| service | non_pos | 9 | kém | 1.04 | 3 | {'non_pos': 3} | đỡ ồn nhưng phím cách vẫn to, đặt lần 1 hết hàng nhưng vẫn để sản phẩm còn, đặt rồi hết hàng không n |
| service | non_pos | 14 | puller | 0.91 | 2 | {'non_pos': 1, 'pos': 1} | sản phẩm tốt, đúng với mô tả. switch dùng rất êm, switch puller hơi khó dùng |

## 6. Mẫu câu sai để đọc tay

`results/error_samples.csv`: 50 comment (ưu tiên lỗi ở lớp non_pos). Đã đọc tay 50/50 comment (cột `nguyen_nhan`, `ghi_chu`, `nguoi_danh_gia`).

Mỗi comment ghi một nguyên nhân chính:

| mã | nguyên nhân | số comment |
|---|---|---|
| A | Diễn đạt hiếm / không có từ khoá quen → bỏ sót hoặc gán nhầm aspect | 18 |
| B | Chê nhẹ nằm trong câu khen (hơi, nhưng, có điều, tiếc…) → đoán pos | 15 |
| C | Nhãn trung tính (điểm 5–6: nêu tính năng, chấp nhận theo giá) bị gộp vào non_pos | 13 |
| D | Không dấu / lỗi gõ | 3 |
| E | Kỳ vọng cá nhân của người mua (quy tắc 42) | 1 |

