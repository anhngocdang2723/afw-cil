from __future__ import annotations

import argparse
import ast
import shutil
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


PARAM_KEYS = [
    "DATASET_NAME",
    "DATASET_FILE",
    "N_SPLITS",
    "N_REPEATS",
    "PSO_PARTICLES",
    "PSO_ITERS",
    "INNER_CV_SPLITS",
    "T_INNER",
    "C",
    "NAMEMETHOD",
    "NAMEFUNCTION",
    "BOUNDS",
    "AFW_AUTHOR_K",
    "AFW_AUTHOR_S1",
    "AFW_AUTHOR_S2",
    "AFW_AUTHOR_S3",
    "AFW_AUTHOR_S4",
]

ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
SYM_RE = re.compile(r"#\s*sym\s*:\s*([A-Za-z_]\w*)", re.IGNORECASE)
IR_IN_NAME_RE = re.compile(r"IR(\d+)pct", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gom ket qua moi folder Test_* thanh 1 file Excel nhieu sheet "
            "(info, cleaned data, summary KB1/2/3 va cac CSV con lai)."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("notebook_modified") / "Experiment",
        help="Thu muc chua cac folder ket qua Test_*.",
    )
    parser.add_argument(
        "--notebook-root",
        type=Path,
        default=Path("notebook_modified"),
        help="Thu muc chua cac notebook de lay thong tin cau hinh.",
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        default=None,
        help=(
            "Thu muc tong hop de luu ban sao tat ca file summary. "
            "Mac dinh: <experiment-root>/_summary_reports"
        ),
    )
    parser.add_argument(
        "--glob",
        default="Test_*",
        help="Mau tim folder ket qua can gom (mac dinh: Test_*).",
    )
    return parser.parse_args()


def safe_literal_eval(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return ""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def normalize_to_text(value: Any) -> str:
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value}"
        return "inf" if value > 0 else "-inf"
    return str(value)


def parse_dataset_name_from_folder(folder_name: str) -> str:
    # Format expected: Test_<dataset>_<ddmmyyyy_hhmmss>
    m = re.match(r"^Test_(.+)_\d{8}_\d{6}$", folder_name)
    if m:
        return m.group(1)
    if folder_name.startswith("Test_"):
        return folder_name[len("Test_") :]
    return folder_name


def normalize_result_prefix(stem: str, dataset_name: str) -> str:
    """Remove noisy prefixes/suffixes like Test_<dataset> and timestamps."""
    text = stem

    prefix = f"Test_{dataset_name}_"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    elif text.startswith("Test_"):
        text = text[len("Test_") :]

    # Remove trailing run timestamp if present: _ddmmyyyy_hhmmss
    text = re.sub(r"_\d{8}_\d{6}$", "", text)
    return text.strip("_")


def find_notebook_for_dataset(dataset_name: str, notebook_root: Path) -> Path | None:
    expected = notebook_root / f"PSO_AFWCIL_QuickTest_{dataset_name}.ipynb"
    if expected.exists():
        return expected

    lowered = dataset_name.lower()
    candidates = sorted(notebook_root.glob("*.ipynb"))
    ranked: list[tuple[int, Path]] = []
    for nb in candidates:
        name_lower = nb.name.lower()
        if lowered not in name_lower:
            continue
        score = 0
        if "quicktest" in name_lower:
            score += 2
        if f"quicktest_{lowered}" in name_lower:
            score += 2
        if "copy" not in name_lower and "base" not in name_lower:
            score += 1
        ranked.append((score, nb))

    if not ranked:
        return None
    ranked.sort(key=lambda t: (t[0], t[1].name.lower()), reverse=True)
    return ranked[0][1]


def parse_notebook_params(nb_path: Path | None) -> dict[str, Any]:
    if nb_path is None or not nb_path.exists():
        return {}

    with nb_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    out: dict[str, Any] = {}
    sym_keys: set[str] = set()

    for cell in obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            lines = source.splitlines()
        else:
            lines = [str(x) for x in source]

        for line in lines:
            sym_match = SYM_RE.search(line)
            if sym_match:
                sym_keys.add(sym_match.group(1).strip())

            code_part = line.split("#", 1)[0].strip()
            if not code_part:
                continue
            assign = ASSIGN_RE.match(code_part)
            if not assign:
                continue

            key = assign.group(1).strip()
            value_text = assign.group(2).strip().rstrip(",")
            if key in PARAM_KEYS or key in sym_keys:
                out[key] = safe_literal_eval(value_text)

    return out


def read_csv_if_exists(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def get_dataset_stats(df_clean: pd.DataFrame) -> dict[str, Any]:
    if "label" in df_clean.columns:
        y = df_clean["label"]
    else:
        y = df_clean.iloc[:, -1]

    total = int(len(y))
    pos = int((y == 1).sum())
    neg = int((y == -1).sum())
    if pos + neg < total:
        # Fallback when labels are not exactly {-1, +1}
        pos = int((y > 0).sum())
        neg = total - pos

    pos_rate = (pos / total) if total else 0.0
    ir_neg_pos = (neg / pos) if pos else float("inf")

    return {
        "total": total,
        "pos": pos,
        "neg": neg,
        "pos_rate": pos_rate,
        "ir_neg_pos": ir_neg_pos,
    }


def extract_kb3_ir_levels(summary_kb3: pd.DataFrame | None, csv_files: list[Path]) -> list[float]:
    levels: set[float] = set()

    if summary_kb3 is not None and "IR_target" in summary_kb3.columns:
        for val in summary_kb3["IR_target"].dropna().tolist():
            try:
                levels.add(float(val))
            except Exception:
                pass

    for path in csv_files:
        m = IR_IN_NAME_RE.search(path.name)
        if m:
            levels.add(float(m.group(1)) / 100.0)

    return sorted(levels, reverse=True)


def sanitize_sheet_name(name: str) -> str:
    bad = set('[]:*?/\\')
    cleaned = "".join("_" if ch in bad else ch for ch in name)
    cleaned = cleaned.strip("'")
    if not cleaned:
        cleaned = "sheet"
    return cleaned[:31]


def unique_sheet_name(base: str, used: set[str]) -> str:
    candidate = sanitize_sheet_name(base)
    if candidate not in used:
        used.add(candidate)
        return candidate

    idx = 2
    while True:
        suffix = f"_{idx}"
        trimmed = candidate[: max(1, 31 - len(suffix))]
        trial = f"{trimmed}{suffix}"
        if trial not in used:
            used.add(trial)
            return trial
        idx += 1


def build_info_sheet(
    folder: Path,
    dataset_name: str,
    notebook_path: Path | None,
    params: dict[str, Any],
    stats: dict[str, Any],
    kb3_levels: list[float],
    kb2_checkpoint_files: list[Path],
    kb3_data_files: list[Path],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    rows.append({"Field": "Dataset name", "Value": dataset_name})
    rows.append({"Field": "Result folder", "Value": str(folder)})
    rows.append(
        {
            "Field": "Summary note",
            "Value": "KB2 checkpoint va KB3 *_data duoc giu trong folder ket qua, khong dua vao workbook summary.",
        }
    )
    rows.append(
        {
            "Field": "Notebook source",
            "Value": str(notebook_path) if notebook_path else "Not found",
        }
    )
    rows.append({"Field": "Total samples", "Value": str(stats["total"])})
    rows.append({"Field": "Positive samples (+1)", "Value": str(stats["pos"])})
    rows.append({"Field": "Negative samples (-1)", "Value": str(stats["neg"])})
    rows.append(
        {"Field": "Positive ratio", "Value": f"{stats['pos_rate']:.6f}"}
    )
    rows.append(
        {
            "Field": "Imbalance ratio IR = neg/pos",
            "Value": (
                f"{stats['ir_neg_pos']:.6f}"
                if math.isfinite(stats["ir_neg_pos"])
                else "inf"
            ),
        }
    )

    rows.append(
        {
            "Field": "KB2 checkpoint files (stored at)",
            "Value": ", ".join(p.name for p in kb2_checkpoint_files)
            if kb2_checkpoint_files
            else "None",
        }
    )
    rows.append(
        {
            "Field": "KB3 data files (stored at)",
            "Value": ", ".join(p.name for p in kb3_data_files)
            if kb3_data_files
            else "None",
        }
    )

    if kb3_levels:
        rows.append(
            {
                "Field": "KB3 IR levels (target)",
                "Value": ", ".join(f"{x:.4f}" for x in kb3_levels),
            }
        )
    else:
        rows.append({"Field": "KB3 IR levels (target)", "Value": ""})

    ordered_params = [k for k in PARAM_KEYS if k in params]
    other_params = sorted([k for k in params.keys() if k not in ordered_params])
    for key in ordered_params + other_params:
        rows.append({"Field": f"config.{key}", "Value": normalize_to_text(params[key])})

    return pd.DataFrame(rows)


def workbook_filename(dataset_name: str) -> str:
    return f"{dataset_name}_summary.xlsx"


def write_workbook(
    out_path: Path,
    dataset_name: str,
    df_info: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_kb1: pd.DataFrame | None,
    df_kb2: pd.DataFrame | None,
    df_kb3: pd.DataFrame | None,
    detail_csv: list[Path],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    used_sheet_names: set[str] = set()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_info.to_excel(
            writer,
            sheet_name=unique_sheet_name("Overview", used_sheet_names),
            index=False,
        )
        df_clean.to_excel(
            writer,
            sheet_name=unique_sheet_name("Dataset_Cleaned", used_sheet_names),
            index=False,
        )

        if df_kb1 is not None:
            df_kb1.to_excel(
                writer,
                sheet_name=unique_sheet_name("KB1_Summary", used_sheet_names),
                index=False,
            )
        if df_kb2 is not None:
            df_kb2.to_excel(
                writer,
                sheet_name=unique_sheet_name("KB2_Summary", used_sheet_names),
                index=False,
            )
        if df_kb3 is not None:
            df_kb3.to_excel(
                writer,
                sheet_name=unique_sheet_name("KB3_Summary", used_sheet_names),
                index=False,
            )

        for path in detail_csv:
            df = pd.read_csv(path)
            cleaned_stem = normalize_result_prefix(path.stem, dataset_name)
            base_sheet = cleaned_stem if cleaned_stem else path.stem
            sheet = unique_sheet_name(base_sheet, used_sheet_names)
            df.to_excel(writer, sheet_name=sheet, index=False)


def export_one_folder(
    folder: Path,
    notebook_root: Path,
    collection_dir: Path,
) -> tuple[Path, Path]:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Khong tim thay CSV trong folder: {folder}")

    dataset_name = parse_dataset_name_from_folder(folder.name)
    notebook_path = find_notebook_for_dataset(dataset_name, notebook_root)
    params = parse_notebook_params(notebook_path)

    cleaned_csv = next((p for p in csv_files if p.name.endswith("_dataset_cleaned.csv")), None)
    if cleaned_csv is None:
        raise FileNotFoundError(f"Khong tim thay file *_dataset_cleaned.csv trong {folder}")

    kb1_summary = next((p for p in csv_files if "KB1_summary" in p.name), None)
    kb2_summary = next((p for p in csv_files if "KB2_summary" in p.name), None)
    kb3_summary = next((p for p in csv_files if "KB3_summary" in p.name), None)

    df_clean = pd.read_csv(cleaned_csv)
    stats = get_dataset_stats(df_clean)

    df_kb1 = read_csv_if_exists(kb1_summary)
    df_kb2 = read_csv_if_exists(kb2_summary)
    df_kb3 = read_csv_if_exists(kb3_summary)

    excluded = {cleaned_csv, kb1_summary, kb2_summary, kb3_summary}
    raw_detail_csv = [p for p in csv_files if p not in excluded]

    kb2_checkpoint_files = [
        p
        for p in raw_detail_csv
        if "KB2" in p.name and "checkpoint" in p.name.lower()
    ]
    kb3_data_files = [
        p
        for p in raw_detail_csv
        if "KB3" in p.name and p.name.lower().endswith("_data.csv")
    ]
    skip_in_workbook = set(kb2_checkpoint_files + kb3_data_files)
    detail_csv = [p for p in raw_detail_csv if p not in skip_in_workbook]

    kb3_levels = extract_kb3_ir_levels(df_kb3, csv_files)
    df_info = build_info_sheet(
        folder,
        dataset_name,
        notebook_path,
        params,
        stats,
        kb3_levels,
        kb2_checkpoint_files,
        kb3_data_files,
    )

    local_out = folder / workbook_filename(dataset_name)
    collection_out = collection_dir / workbook_filename(dataset_name)

    try:
        write_workbook(
            local_out,
            dataset_name,
            df_info,
            df_clean,
            df_kb1,
            df_kb2,
            df_kb3,
            detail_csv,
        )

        collection_out.parent.mkdir(parents=True, exist_ok=True)
        if collection_out.resolve() != local_out.resolve():
            shutil.copy2(local_out, collection_out)
    except ImportError as exc:
        raise RuntimeError(
            "Khong ghi duoc file .xlsx vi thieu thu vien openpyxl. "
            "Hay cai dat bang lenh: pip install openpyxl"
        ) from exc

    return local_out, collection_out


def main() -> None:
    args = parse_args()
    exp_root = args.experiment_root
    nb_root = args.notebook_root
    collection_dir = args.collection_dir or (exp_root / "_summary_reports")

    if not exp_root.exists():
        raise FileNotFoundError(f"Khong ton tai experiment root: {exp_root}")
    if not nb_root.exists():
        raise FileNotFoundError(f"Khong ton tai notebook root: {nb_root}")

    folders = sorted([p for p in exp_root.glob(args.glob) if p.is_dir()])
    if not folders:
        print(f"Khong tim thay folder nao theo mau '{args.glob}' trong: {exp_root}")
        return

    print(f"Tim thay {len(folders)} folder ket qua.")
    for idx, folder in enumerate(folders, start=1):
        try:
            local_file, central_file = export_one_folder(folder, nb_root, collection_dir)
            print(f"[{idx}/{len(folders)}] OK  -> local: {local_file}")
            print(f"[{idx}/{len(folders)}] OK  -> central: {central_file}")
        except Exception as exc:
            print(f"[{idx}/{len(folders)}] FAIL -> {folder} | {exc}")


if __name__ == "__main__":
    main()
