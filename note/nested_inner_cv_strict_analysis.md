# Phân tích triển khai strict train/validation/test cho PSO-AFW-CIL

## Mục tiêu
Chuẩn hoá pipeline theo nguyên tắc khoa học:
1. Không dùng outer test trong quá trình chọn tham số.
2. Không để dữ liệu SMOTE lọt vào validation/test.
3. Tách rõ các pha: tuning (core/val) và báo cáo cuối (outer test).

## Các thay đổi logic đã áp dụng trong notebook
File: `notebook_modified/PSO_AFWCIL_QuickTest_haberman copy 2.ipynb`

### 1) Tách core/validation không chồng lấn
- Thêm hàm `split_train_core_val(X_train, y_train, ...)`.
- Hàm trả về `X_core, y_core, X_val, y_val` bằng `train_test_split(..., stratify=y_train)`.
- Nếu split lỗi (mẫu quá ít), fallback an toàn để không crash.

Ý nghĩa:
- Validation không còn nằm trong tập train dùng để fit mô hình khi tuning.
- Khắc phục lỗ hổng overlap train-val của phiên bản trước.

### 2) Tuning dùng core/val, không dùng outer test
- KB1/KB2/KB3 đều chuyển sang:
  - Tuning params: dùng `X_core, y_core` (train) và `X_val_tune, y_val_tune` (validate).
  - Outer test `X_te, y_te` chỉ dùng khi đánh giá cuối.

### 3) SMOTE chỉ dùng trên train cho tuning
- Với pipeline SMOTE + PSO:
  - Tạo `X_core_sm, y_core_sm` từ `X_core, y_core` để tuning.
  - Validation vẫn là dữ liệu thật (`X_val_tune, y_val_tune`).
- Không dùng dữ liệu SMOTE để chấm validation/test.

### 4) Refit final trên full outer-train sau khi chọn tham số
- Thêm hàm `lfb_refit_with_params(...)` để train lại AFW trên full outer-train (hoặc full outer-train đã SMOTE) với bộ tham số đã chọn.
- Quy trình mới:
  1. Chọn params bằng core/val.
  2. Refit trên full-train.
  3. Test outer fold đúng 1 lần.

Điểm mạnh:
- Dùng nhiều dữ liệu train hơn cho mô hình cuối.
- Vẫn giữ tách biệt test khỏi pha tuning.

## Mapping theo checklist phương pháp luận
1. Tách Validation: ✅
2. SMOTE chỉ trên train lõi khi tuning: ✅
3. PSO tối ưu trên validation: ✅
4. Refit và test cuối đúng 1 lần: ✅

## Cảnh báo còn lại
- Diagnostics hiện chỉ còn cảnh báo không chặn chạy: import `tracemalloc` chưa dùng.
- Không ảnh hưởng đến kết quả thực nghiệm.

## Gợi ý nâng cấp tiếp theo (nếu cần rigor cao hơn)
- Thay inner holdout bằng inner K-fold đầy đủ để thành nested CV chuẩn nghiêm ngặt.
- Trong nested đầy đủ:
  - Inner score = mean/std trên các inner folds.
  - Outer score = mean/std trên outer folds dùng để báo cáo chính thức.

## Kết luận
Notebook hiện đã đi đúng logic chống leakage cho thực nghiệm khoa học:
- Tuning độc lập với test,
- dữ liệu tổng hợp không rò sang validation/test,
- và đánh giá cuối phản ánh năng lực tổng quát hoá thực tế hơn.