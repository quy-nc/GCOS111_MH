# Error analysis (bước 5)

Model B1-SVM trên test (563 comment). Comment sai ít nhất 1 aspect: 334 (59%); tổng số ô (comment × aspect) sai: 525/4504.

## 1. Loại lỗi theo aspect

| aspect | ô sai | bỏ sót (nhắc → none) | báo nhầm (none → có) | pos → non_pos | non_pos → pos |
|---|---|---|---|---|---|
| sound | 67 | 34 | 23 | 5 | 5 |
| typing | 91 | 50 | 30 | 3 | 8 |
| quantity | 153 | 78 | 61 | 7 | 7 |
| lighting | 37 | 21 | 1 | 0 | 15 |
| connectivity | 26 | 16 | 1 | 5 | 4 |
| price | 52 | 33 | 11 | 1 | 7 |
| delivery | 34 | 21 | 7 | 1 | 5 |
| service | 65 | 31 | 26 | 5 | 3 |
| tổng | 525 | 284 | 160 | 27 | 54 |

## 2. Tỷ lệ lỗi theo đặc điểm comment

Nhóm không loại trừ nhau (một comment có thể vừa dài vừa có phủ định).

| nhóm | số comment | % comment sai ≥1 aspect | số ô sai / comment |
|---|---|---|---|
| toàn bộ test | 563 | 59.3 | 0.93 |
| có phủ định (không/chẳng/chưa) | 162 | 66.0 | 1.2 |
| có đối lập (nhưng/tuy/mỗi tội/tiếc…) | 88 | 68.2 | 1.18 |
| không dấu / telex | 9 | 77.8 | 1.0 |
| ngắn (≤ 8 từ) | 51 | 41.2 | 0.51 |
| trung bình (9–25 từ) | 367 | 58.3 | 0.82 |
| dài (> 25 từ) | 145 | 68.3 | 1.36 |
| nhắc 0–1 aspect | 167 | 44.3 | 0.56 |
| nhắc 2–3 aspect | 316 | 60.8 | 0.9 |
| nhắc ≥ 4 aspect | 80 | 85.0 | 1.82 |

### Chữ "êm" với sound / typing

| aspect | nhóm | số comment | % sai |
|---|---|---|---|
| sound | có 'êm' | 93 | 24.7 |
| sound | không có 'êm' | 470 | 9.4 |
| typing | có 'êm' | 93 | 32.3 |
| typing | không có 'êm' | 470 | 13.0 |

## 3. Theo mẫu bàn phím (≥ 20 comment test)

| model | số comment | % đúng cả 8 aspect | detection_f1 | mean_macro_f1 (n nhỏ, tham khảo) |
|---|---|---|---|---|
| AULA F75 MAX | 36 | 25.0 | 0.772 | 0.458 |
| ATAS ZA981 V4 | 49 | 30.6 | 0.8 | 0.564 |
| WY - 68K | 37 | 32.4 | 0.715 | 0.549 |
| F68 | 93 | 34.4 | 0.812 | 0.596 |
| AULA F75 | 63 | 34.9 | 0.803 | 0.616 |
| AULA S75 PRO | 49 | 36.7 | 0.813 | 0.594 |
| K550 V9/K200 | 38 | 39.5 | 0.844 | 0.538 |
| MAD68/MAD60 | 56 | 46.4 | 0.829 | 0.545 |
| K200 | 100 | 63.0 | 0.867 | 0.629 |

## 4. SVM và E1 Rule-based bổ sung cho nhau

Số ô (comment × aspect). Nếu mỗi ô chọn được model đúng (oracle, không làm được trong thực tế) thì mean_macro_f1 = 0.792, so với SVM 0.627 và E1 0.643 → trần trên của việc kết hợp hai cách.

| aspect | cả hai đúng | chỉ SVM đúng | chỉ E1 đúng | cả hai sai |
|---|---|---|---|---|
| sound | 458 | 38 | 36 | 31 |
| typing | 423 | 49 | 41 | 50 |
| quantity | 261 | 149 | 82 | 71 |
| lighting | 503 | 23 | 22 | 15 |
| connectivity | 520 | 17 | 14 | 12 |
| price | 473 | 38 | 22 | 30 |
| delivery | 515 | 14 | 18 | 16 |
| service | 443 | 55 | 28 | 37 |
| tổng | 3596 | 383 | 263 | 262 |

## 5. Từ trong top 15 trọng số của SVM nhưng chỉ có trong ≤ 3 comment train

35 từ. Đây là dấu hiệu model học thuộc vài comment cụ thể thay vì quy luật chung.

| aspect | lớp | hạng | từ | trọng số | số comment train | nhãn aspect trong các comment đó | ví dụ |
|---|---|---|---|---|---|---|---|
| sound | pos | 11 | chất_âm | 1.6 | 3 | {'pos': 2, 'none': 1} | bàn phím giao nhanh chất âm tốt có đầy đủ hướng dẫn sử dụng 826 |
| sound | non_pos | 6 | câu | 1.81 | 2 | {'non_pos': 1, 'none': 1} | độ bền chắc tay. ánh sáng tốt. phím cơ ồn vl nhưng đầm tay. sản phẩm tốt nhưng bên hỗ trợ thường hay |
| sound | non_pos | 7 | ồn_chút | 1.78 | 2 | {'non_pos': 2} | ổn. khá là nặng,gõ cũng ổn. hơi ồn chút bọc hàng cẩn thận. mua được nha |
| typing | pos | 10 | nhanh_phím | 1.88 | 2 | {'pos': 2} | tốt. giao hàng nhanh phím rất nhạy kết nối nhanh nữa |
| typing | non_pos | 1 | trưng_bày | 2.74 | 2 | {'none': 1, 'non_pos': 1} | tot. êm. bàn phím rất đẹp nha, nhỏ gọn xinh xinh, mua về không nỡ dùng cho vào hộp trưng bày |
| typing | non_pos | 3 | soạn_thảo | 2.23 | 2 | {'non_pos': 1, 'none': 1} | tốt. nhiều tính năng cho soạn thảo. bàn phím đẹp, gõ ổn |
| typing | non_pos | 6 | mom | 1.98 | 3 | {'non_pos': 2, 'none': 1} | màu ở ngoài xinh hơn trên hình á các mom, gõ êm, phù hợp cho dân mới chơi, nhưng mà phím nhạy lắm nê |
| typing | non_pos | 10 | tiêp | 1.63 | 2 | {'pos': 1, 'non_pos': 1} | shop giao hàng rất nhanh, bàn phím bấm êm, rất đẹp. sẽ ủng hộ tiêp |
| typing | non_pos | 13 | trượt | 1.51 | 3 | {'none': 2, 'non_pos': 1} | phím đẹp mà nuột lắm nha. nhảy au thì không trượt phát nào. còn được tặng kèm cái lót chuột.ãn nguyệ |
| quantity | non_pos | 11 | k780 | 1.78 | 2 | {'non_pos': 1, 'none': 1} | ổn. giao khá nhanh, có 78 key thôi, nói thật thì thích za981 hơn mà t lỡ mua nhầm k780 |
| lighting | pos | 9 | nhạc | 1.88 | 3 | {'pos': 3} | ngon như giá. leg theo nhạc emoji_pos. ngon |
| lighting | pos | 10 | led_ok | 1.77 | 2 | {'pos': 2} | giống như mô tả hình ảnh, bàn phím đẹp,led ok, không bị lỗi hay bất cứ gì hêts |
| lighting | pos | 11 | ngon_led | 1.63 | 2 | {'pos': 2} | ngon. có có. phím ngon led xịn bấm êm tai nói chung kho có gì để chê |
| lighting | non_pos | 1 | led_ổn | 3.46 | 3 | {'non_pos': 3} | độ nhạy tốt. thiết kế đẹp led ổn. chất liệu nhựa tốt. kt phím không bị lỗi còn kết đủ loại thiết bị |
| lighting | non_pos | 3 | cố_định | 2.58 | 3 | {'non_pos': 3} | độ bền tốt. ánh sáng bình thường thôi cái này là led rgb cố định. dùng được 4 ngày rồi phím bị đổ bi |
| lighting | non_pos | 6 | hi | 1.9 | 3 | {'non_pos': 2, 'none': 1} | tuyệt vời. tốt, bàn phím tuyệt vời gõ êm,mỗi cái key cap tối nên muốn màu khác xuyên led hi |
| lighting | non_pos | 15 | xịn_sò | 1.5 | 2 | {'non_pos': 1, 'none': 1} | bàn phím nhìn xịn sò lắm, không có gi chê, đánh êm tay không ồn, đi chung quả đèn chớp chớp như dj |
| connectivity | pos | 3 | pink | 2.26 | 2 | {'pos': 2} | âm thanh pink switch, tiếng nhỏ. chất lượng đầy đủ tính năng dù chữ trên keycap chưa được đẹp lắm |
| connectivity | pos | 4 | nhanh_phím | 2.2 | 2 | {'pos': 1, 'none': 1} | tốt. giao hàng nhanh phím rất nhạy kết nối nhanh nữa |
| connectivity | pos | 5 | swith | 2.14 | 2 | {'pos': 1, 'none': 1} | độ nhạy ngon, không bi delay. thiết kế lộ swith nhưng vẫn đẹp |
| connectivity | pos | 6 | bền_pin | 2.05 | 2 | {'pos': 2} | ok. quá ưng quá đẹp quá bền pin cũng ok đầy đủ phụ kiện âm trầm ấm êm tai |
| connectivity | pos | 9 | gon | 1.83 | 2 | {'none': 1, 'pos': 1} | độ nhạy nhay. thiết kế gon. chất liệu ben. gõ rất êm tay, bàn phím lại gọn vừa đủ nhu cầu sử dụng gi |
| connectivity | pos | 12 | hạt | 1.59 | 2 | {'pos': 1, 'none': 1} | bifnhb thuong. cầm chắc tay. giá hạt rẻ dây có thể đi gọn, cầm phím không lọc xọc mong là bền |
| connectivity | non_pos | 4 | chập_chờn | 2.55 | 3 | {'non_pos': 3} | chất lượng không tốt. ngoại quan đẹp, nút cách hơi không được nhậy, còn lại các nút đều bình thường. |
| connectivity | non_pos | 6 | dây_cáp | 1.94 | 2 | {'non_pos': 2} | cũng tốt. có led nhiều màu, núm xoay tăng giảm âm lượng và độ sáng đèn bàn phím. h mới có thể đánh g |
| connectivity | non_pos | 7 | wireless | 1.93 | 2 | {'non_pos': 2} | tốt, key không có xuyên led. rgb led, lcd screen, wireless, về thay keycap xuyên led là đầy đủ tính  |
| connectivity | non_pos | 9 | cố_định | 1.65 | 3 | {'none': 2, 'non_pos': 1} | độ bền tốt. ánh sáng bình thường thôi cái này là led rgb cố định. dùng được 4 ngày rồi phím bị đổ bi |
| connectivity | non_pos | 14 | day | 1.32 | 3 | {'none': 2, 'non_pos': 1} | không day. chat luong tot. giao hàng nhanh nhỏ gọn chất lương khá ok |
| price | non_pos | 7 | cos | 1.73 | 3 | {'pos': 1, 'none': 1, 'non_pos': 1} | phism ok gias rer nen khong cos camr giasc gox thichs. tạm. rẻ ok phù hợp túi tiền |
| price | non_pos | 8 | hoàn | 1.73 | 3 | {'none': 2, 'non_pos': 1} | phím goc kêu tiếng to, cân nhắc khi mua hàng, mua không hoàn lại được, khá tệ hại |
| delivery | pos | 12 | giao_hơi | 1.46 | 3 | {'pos': 3} | sản phẩm tuyệt vời lắm ạ, đáng các bạn sinh viên kham thảo nhé, có hơi mùi mũ 1 xíu nhưng mà không s |
| delivery | non_pos | 9 | biet | 1.41 | 3 | {'none': 2, 'non_pos': 1} | khong biet. khong biet. tui mua cho ny nên không biết đánh giá gì nữa hihihi |
| service | pos | 14 | nhắn_shop | 1.66 | 3 | {'pos': 3} | rẻ, ngon. tốt. mua về tập mod or sử dụng cơ bản ngon luôn nhée, với yêu shop lắm nha nhắn shop giao  |
| service | non_pos | 7 | puller | 1.94 | 2 | {'non_pos': 1, 'pos': 1} | sản phẩm tốt, đúng với mô tả. switch dùng rất êm, switch puller hơi khó dùng |
| service | non_pos | 8 | 30 | 1.92 | 3 | {'non_pos': 1, 'none': 1, 'pos': 1} | sản phẩm đóng gói cẩn thận, có chế độ bảo hành 30 ngày bảo hành 237 |

## 6. Mẫu câu sai để đọc tay

`results/error_samples.csv`: 50 comment (ưu tiên lỗi ở lớp non_pos). Đã đọc tay 13/50 comment (cột `nguyen_nhan`, `ghi_chu`, `nguoi_danh_gia`).

Mỗi comment ghi một nguyên nhân chính:

| mã | nguyên nhân | số comment |
|---|---|---|
| A | Diễn đạt hiếm / không có từ khoá quen → bỏ sót hoặc gán nhầm aspect | 4 |
| B | Chê nhẹ nằm trong câu khen (hơi, nhưng, có điều, tiếc…) → đoán pos | 2 |
| C | Nhãn trung tính (điểm 5–6: nêu tính năng, chấp nhận theo giá) bị gộp vào non_pos | 5 |
| D | Không dấu / lỗi gõ | 2 |
| E | Kỳ vọng cá nhân của người mua (quy tắc 42) | 0 |

