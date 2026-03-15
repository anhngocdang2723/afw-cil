# Train-only tuning cho AFW-CIL/PSO/GridSearch

## 1) Vấn đề phương pháp luận trước khi sửa
Trong cấu hình cũ, các bước tối ưu tham số/weight (AFW, GridSearch, PSO) dùng trực tiếp `X_te, y_te` của outer fold để chấm điểm trong quá trình tối ưu.

Hệ quả:
- Có **test leakage** (mô hình "nhìn" nhãn test khi chọn tham số).
- Kết quả outer-CV có xu hướng lạc quan hơn thực tế.
- Không đúng tinh thần đánh giá khoa học: test set phải là dữ liệu "chưa từng được dùng" cho lựa chọn mô hình.

## 2) Nguyên tắc đúng trong nghiên cứu
Với CV chuẩn, dữ liệu trong mỗi outer fold nên tách vai trò:
- **Outer-train**: dùng để học + tuning.
- **Outer-test**: chỉ dùng 1 lần để đánh giá cuối.

Tóm tắt công thức mục tiêu tuning:

- Chọn tham số $\theta$ để tối đa hoá metric trên validation nội bộ:

$$
\theta^* = \arg\max_{\theta} \; \text{Score}(\mathcal{D}_{train\_inner}, \mathcal{D}_{val\_inner}; \theta)
$$

- Sau đó cố định $\theta^*$, huấn luyện trên outer-train và chỉ đánh giá trên outer-test.

## 3) Cách đã triển khai trong notebook
File đã cập nhật:
- `notebook_modified/PSO_AFWCIL_QuickTest_haberman copy 2.ipynb`

Thay đổi chính:
1. Thêm helper `make_tuning_validation(X_train, y_train, ...)` để tạo validation nội bộ từ train fold.
2. Trong KB1/KB2/KB3:
   - Tạo `X_val_tune, y_val_tune` từ `X_tr, y_tr`.
   - Tất cả bước tối ưu (AFW `lfb_fixed`, GridSearch `grid_search_afwcil`, PSO `PSO_AFWCIL`) dùng `X_val_tune, y_val_tune`.
   - Outer test (`X_te, y_te`) chỉ dùng để tính metric cuối và ghi CSV.
3. Quick test cell cuối cũng đổi sang train-only tuning (PSO dùng `X_val_q, y_val_q` để optimize).

## 4) Vì sao cách này hợp lý hơn về khoa học
- Giảm thiên lệch đánh giá (optimistic bias).
- Tách rõ hai mục tiêu:
  - Tuning/selection trên dữ liệu train.
  - Generalization check trên dữ liệu test chưa dùng.
- Báo cáo kết quả có độ tin cậy cao hơn khi so sánh mô hình.

## 5) Lưu ý thực nghiệm
- Hiện tại đang dùng **inner holdout** (validation tách từ outer-train) để cân bằng giữa độ chuẩn và thời gian chạy.
- Nếu muốn chuẩn hơn nữa theo nested-CV đầy đủ, có thể thay inner holdout bằng inner K-fold (tốn thời gian hơn đáng kể).

## 6) Cách mô tả trong báo cáo/luận văn
Bạn có thể mô tả ngắn gọn:

> Trong mỗi outer fold, chúng tôi chỉ sử dụng phần train để tối ưu AFW/PSO/GridSearch thông qua một validation nội bộ. Phần test của outer fold không tham gia tuning và chỉ được dùng để đánh giá hiệu năng cuối cùng.

Câu này giúp tránh phản biện về leakage và làm rõ tính hợp lệ của protocol đánh giá.