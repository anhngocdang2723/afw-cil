# Golden Setup cho Paper: Nested Inner-CV trong PSO

## Mục tiêu
Thiết lập cấu hình cân bằng giữa **rigor học thuật cao** và **thời gian chạy hợp lý** cho bộ thí nghiệm AFW/PSO:

- Outer CV: `5-fold`
- Inner CV (chỉ trong fitness của PSO): `3-fold`
- PSO: `particles=8`, `iters=10`
- SMOTE: chỉ áp dụng trên **inner-train** (không áp dụng lên inner-validation)

## Cấu hình đã triển khai
Trong notebook `notebook_modified/PSO_AFWCIL_QuickTest_haberman copy 2.ipynb`:

1. Cập nhật tham số mặc định:
   - `PSO_PARTICLES = 8`
   - `PSO_ITERS = 10`
   - `INNER_CV_SPLITS = 3`
   - `N_SPLITS = 5` giữ nguyên cho outer CV

2. Refactor lớp `PSO_AFWCIL`:
   - Thêm tham số:
     - `inner_cv_splits=3`
     - `use_smote_inner=False`
   - `fitness_function` giờ đánh giá mỗi particle bằng:
     - `StratifiedKFold(n_splits=inner_cv_splits)` trên dữ liệu outer-train/core
     - Mỗi inner fold tính `G-mean`
     - Trả về `mean(G-mean)` của các inner fold

3. Quy tắc SMOTE trong inner CV:
   - Nếu `use_smote_inner=True`: SMOTE chỉ fit/resample trên `X_inner_train, y_inner_train`
   - Inner-validation (`X_inner_val, y_inner_val`) luôn giữ nguyên phân phối tự nhiên

4. Mapping theo kịch bản:
   - PSO-AFW-CIL (không SMOTE): `use_smote_inner=False`
   - BL-SMOTE + PSO-AFW-CIL: `use_smote_inner=True`
   - Final model vẫn refit trên full outer-train theo logic của từng nhánh (SMOTE hoặc non-SMOTE), rồi mới đánh giá outer-test 1 lần.

## Ý nghĩa học thuật
Thiết kế này đảm bảo:

- Tách bạch tuning và testing ở outer loop
- Giảm phụ thuộc vào một lần inner split (so với holdout)
- Hạn chế optimistic bias khi chọn tham số
- Giữ tính tái lập và cân bằng chi phí tính toán

## Ghi chú runtime
So với inner holdout, nested inner 3-fold làm tăng chi phí tuning gần xấp xỉ theo hệ số ~3 cho phần fitness PSO, nhưng đã được bù bằng:

- giảm số hạt xuống 8
- giảm số vòng PSO xuống 10

Đây là cấu hình “golden” phù hợp để báo cáo cho paper trên dataset vừa/lớn.
