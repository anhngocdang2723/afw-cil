# Dataset Switch Note: Transfusion & Yeast

Đã tạo 2 notebook mới từ bản golden setup nested inner-CV:

1. `notebook_modified/PSO_AFWCIL_QuickTest_transfusion.ipynb`
2. `notebook_modified/PSO_AFWCIL_QuickTest_yeast.ipynb`

## 1) Transfusion notebook
- `DATASET_NAME = "transfusion"`
- `DATASET_FILE = "/home/quangvd/project/FAIR-2022/data/datasets/transfusion.csv"`
- `LABEL_COL = "whether he/she donated blood in March 2007"`
- `LABEL_MAP = {1: 1.0, 0: -1.0}`

## 2) Yeast notebook
Pipeline hiện tại là nhị phân, nên yeast được cấu hình theo one-vs-rest:
- `DATASET_NAME = "yeast_mit_vs_rest"`
- `DATASET_FILE = "/home/quangvd/project/FAIR-2022/data/datasets/yeast.csv"`
- `LABEL_COL = "name"`
- `LABEL_MAP` đặt `MIT -> +1`, các lớp còn lại -> `-1`

## Lưu ý chạy test
- Các notebook kế thừa full cấu hình Golden Setup trước đó:
  - outer CV = 5
  - inner CV (PSO fitness) = 3
  - PSO particles = 8, iters = 10
  - SMOTE chỉ dùng trên inner-train khi `use_smote_inner=True`
- Đường dẫn dataset đã dùng **absolute path** để tránh lỗi môi trường làm việc.
