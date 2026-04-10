import pandas as pd
import glob
import os
import re

def process_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục không tồn tại: {folder_path}")
        return

    # Tìm tất cả file summary trong thư mục
    summary_files = glob.glob(os.path.join(folder_path, "*_summary_*.csv"))
    # Loại trừ các file đã được extended trước đó nếu lỡ chạy lại 2 lần
    summary_files = [f for f in summary_files if "_extended." not in f]

    if not summary_files:
        print("⚠️ Không tìm thấy file có đuôi '*_summary_*.csv' nào trong thư mục!")
        return

    metrics = ["SP", "SE", "Gmean", "F1 Score", "Accuracy", "AUC"]
    method_map = {
        "W-SVM": "WSVM",
        "AFW-CIL_default": "AFW",
        "BL-SMOTE+PSO-AFW-CIL": "Proposed"
    }

    for summary_file in summary_files:
        summary_filename = os.path.basename(summary_file)
        print(f"\n⏳ Đang xử lý file tổng hợp: {summary_filename}")
        
        # Tách phần prefix và timestamp từ tên file
        # VD: Test_abalone_KB3_summary_17032026_214938.csv
        # -> prefix = Test_abalone_KB3, timestamp = 17032026_214938
        match = re.match(r"(.*)_summary_(.*)\.csv", summary_filename)
        if not match:
            print(f"⚠️ Không thể trích xuất prefix/thời gian từ {summary_filename}, bỏ qua.")
            continue
        
        prefix = match.group(1)
        timestamp = match.group(2)

        # Tìm các file chạy chi tiết tương ứng.
        # Ở đây anh dùng wildcard linh hoạt ở phần timestamp vì em có thể chạy bù file con ở các thời điểm khác nhau.
        detail_pattern = os.path.join(folder_path, f"{prefix}_IR*pct_*.csv")
        detail_files = glob.glob(detail_pattern)
        
        # Lọc bỏ các file dataset lưu kèm
        detail_files = [f for f in detail_files if "_data.csv" not in f and "_summary_" not in f]

        if not detail_files:
            print(f"⚠️ Không tìm thấy file folder con (mẫu: {prefix}_IR...pct_{timestamp}) cho {summary_filename}.")
            continue

        all_rows = []
        for f in detail_files:
            fname = os.path.basename(f)
            try:
                # Trích xuất giá trị % từ tên file
                ir_str = re.search(r"_IR(\d+)pct", fname).group(1)
                ir_val = float(ir_str) / 100.0
            except Exception as e:
                print(f"⚠️ Không tự động nhận diện được tỉ lệ lớp % trong tên file {fname}, bỏ qua.")
                continue
            
            df = pd.read_csv(f)
            df["IR_target"] = round(ir_val, 2)
            all_rows.append(df)

            if all_rows:
                all_data = pd.concat(all_rows, ignore_index=True)
                summary_df = pd.read_csv(summary_file)
                
                # Tính toán mean và std
                grouped = all_data.groupby(["IR_target", "Name Method"])[metrics].agg(["mean", "std"])
                grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
                grouped = grouped.reset_index()
                
                # Gán vào summary main
                new_summary = summary_df.copy()
                if "IR_target" in new_summary.columns:
                    new_summary["IR_target"] = new_summary["IR_target"].round(2)
                
                # Xóa các cột độ đo cũ đã lưu trong file summary để tránh bị lặp (ví dụ: Proposed_SP, W-SVM_SP...)
                cols_to_drop = []
                for col in new_summary.columns:
                    for method_orig, method_short in method_map.items():
                        if col.startswith(f"{method_orig}_") or col.startswith(f"{method_short}_"):
                            cols_to_drop.append(col)
                new_summary = new_summary.drop(columns=list(set(cols_to_drop)))
                
                for method_orig, method_short in method_map.items():
                    method_data = grouped[grouped["Name Method"] == method_orig].copy()
                    if method_data.empty:
                        continue  # Thuật toán này không tồn tại trong data
                    
                    method_data = method_data.drop(columns=["Name Method"])
                    
                    # Đổi tên cho đúng chuẩn: Phương_Pháp_TênĐộĐo_mean/std
                    rename_dict = {col: f"{method_short}_{col}" for col in method_data.columns if col != "IR_target"}
                    method_data = method_data.rename(columns=rename_dict)
                    
                    new_summary = pd.merge(new_summary, method_data, on="IR_target", how="left")
            
            # Lưu ra file mới
            out_file = summary_file.replace(".csv", "_extended.csv")
            new_summary.to_csv(out_file, index=False)
            print(f"✅ Thành công! Đã lưu: {os.path.basename(out_file)}")

if __name__ == '__main__':
    print("="*60)
    print(" CÔNG CỤ TỔNG HỢP VÀ TÍNH TRUNG BÌNH KẾT QUẢ K-FOLD (BATCH)")
    print("="*60)
    while True:
        folder_input = input("\n👉 Hãy nhập đường dẫn thư mục chứa kết quả (vd: D:\\...\\Experiment\\Test_abc)\nHoặc gõ 'q' để thoát: ").strip()
        
        if folder_input.lower() == 'q':
            print("Đã thoát!")
            break
        
        # Bỏ đi dấu nháy kép (nếu khi copy as path trong máy bị dính)
        folder_input = folder_input.strip('\"\'')
        
        if folder_input:
            process_folder(folder_input)
