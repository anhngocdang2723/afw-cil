from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


FOLDER_RE = re.compile(r"^Test_(.+?)_(\d{8})_(\d{6})$")

DOMAIN_MAP: dict[str, str] = {
    "abalone": "Sinh hoc bien",
    "company_bankruptcy_rediction": "Tai chinh",
    "company_bankruptcy_prediction": "Tai chinh",
    "glass1": "Vat lieu",
    "haberman": "Y te",
    "pima": "Y te",
    "transfusion": "Y te",
    "vehicle1": "Thi giac may tinh",
    "vehicle3": "Thi giac may tinh",
    "wisconsin": "Y te",
    "yeast": "Tin sinh hoc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tao file _overview.xlsx gom thong tin dataset tu cac folder Test_* "
            "trong Experiment."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("notebook_modified") / "Experiment",
        help="Thu muc chua cac folder ket qua Test_*.",
    )
    parser.add_argument(
        "--glob",
        default="Test_*",
        help="Mau tim folder ket qua can tong hop (mac dinh: Test_*).",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Thu muc luu file _overview.xlsx (mac dinh: <experiment-root>/_summary_reports).",
    )
    parser.add_argument(
        "--output-name",
        default="_overview.xlsx",
        help="Ten file output (mac dinh: _overview.xlsx).",
    )
    return parser.parse_args()


def parse_dataset_name_from_folder(folder_name: str) -> str:
    match = FOLDER_RE.match(folder_name)
    if match:
        return match.group(1)
    if folder_name.startswith("Test_"):
        return folder_name[len("Test_") :]
    return folder_name


def parse_run_timestamp(folder_name: str, folder_path: Path) -> datetime:
    match = FOLDER_RE.match(folder_name)
    if match:
        date_text = match.group(2)
        time_text = match.group(3)
        try:
            return datetime.strptime(f"{date_text}{time_text}", "%d%m%Y%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(folder_path.stat().st_mtime)


def infer_domain(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized in DOMAIN_MAP:
        return DOMAIN_MAP[normalized]

    if "bankruptcy" in normalized:
        return "Tai chinh"
    if any(key in normalized for key in ["haberman", "pima", "transfusion", "wisconsin"]):
        return "Y te"
    if "vehicle" in normalized:
        return "Thi giac may tinh"
    if "yeast" in normalized:
        return "Tin sinh hoc"
    if "abalone" in normalized:
        return "Sinh hoc bien"

    return "Chua phan loai"


def find_cleaned_csv(folder: Path) -> Path | None:
    candidates = sorted(folder.glob("*_dataset_cleaned.csv"))
    if candidates:
        return candidates[0]

    # Fallback: looser match in case naming drifts.
    for path in sorted(folder.glob("*.csv")):
        if "dataset_cleaned" in path.name:
            return path
    return None


def get_label_series(df: pd.DataFrame) -> pd.Series:
    if "label" in df.columns:
        return df["label"]
    return df.iloc[:, -1]


def compute_stats(df: pd.DataFrame) -> tuple[int, int, int, float]:
    labels = pd.to_numeric(get_label_series(df), errors="coerce")
    total = int(len(labels))

    neg = int((labels == -1).sum())
    pos = int((labels == 1).sum())

    if neg + pos < total:
        pos = int((labels > 0).sum())
        neg = total - pos

    majority = max(neg, pos)
    minority = min(neg, pos)
    ir_pct = (minority / majority * 100.0) if majority else 0.0
    return total, neg, pos, ir_pct


def count_features(df: pd.DataFrame) -> int:
    if "label" in df.columns:
        return max(0, df.shape[1] - 1)
    return max(0, df.shape[1] - 1)


def build_overview_rows(experiment_root: Path, folder_glob: str) -> list[dict[str, object]]:
    folders = sorted([p for p in experiment_root.glob(folder_glob) if p.is_dir()])
    if not folders:
        return []

    # Keep only the latest run for each dataset.
    latest_by_dataset: dict[str, tuple[datetime, Path]] = {}
    for folder in folders:
        dataset_name = parse_dataset_name_from_folder(folder.name)
        run_time = parse_run_timestamp(folder.name, folder)
        current = latest_by_dataset.get(dataset_name)
        if current is None or run_time > current[0]:
            latest_by_dataset[dataset_name] = (run_time, folder)

    rows: list[dict[str, object]] = []
    for dataset_name, (_, folder) in sorted(latest_by_dataset.items(), key=lambda t: t[0].lower()):
        cleaned_csv = find_cleaned_csv(folder)
        if cleaned_csv is None:
            print(f"[WARN] Bo qua {folder.name}: khong tim thay *_dataset_cleaned.csv")
            continue

        df = pd.read_csv(cleaned_csv)
        total, neg, pos, ir_pct = compute_stats(df)
        n_features = count_features(df)

        rows.append(
            {
                "Ten Dataset": dataset_name,
                "Linh vuc": infer_domain(dataset_name),
                "So thuoc tinh": n_features,
                "Tong so mau": total,
                "Mau nhan am (-1)": neg,
                "Mau nhan duong (+1)": pos,
                "Ti le mat can bang (IR%)": round(ir_pct, 4),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root
    summary_dir = args.summary_dir or (experiment_root / "_summary_reports")

    if not experiment_root.exists():
        raise FileNotFoundError(f"Khong ton tai experiment root: {experiment_root}")

    rows = build_overview_rows(experiment_root, args.glob)
    if not rows:
        print(f"Khong tim thay du lieu hop le theo mau '{args.glob}' trong {experiment_root}")
        return

    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = summary_dir / args.output_name

    df_overview = pd.DataFrame(rows)
    df_overview = df_overview.sort_values(by=["Ten Dataset"]).reset_index(drop=True)

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_overview.to_excel(writer, sheet_name="Overview", index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Khong ghi duoc file .xlsx vi thieu thu vien openpyxl. "
            "Hay cai dat bang lenh: pip install openpyxl"
        ) from exc

    print(f"Da tao file overview: {output_path}")
    print(f"Tong so dataset: {len(df_overview)}")


if __name__ == "__main__":
    main()
