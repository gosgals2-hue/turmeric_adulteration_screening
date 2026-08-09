"""
validate_dataset.py — Audit collected turmeric dataset before any modeling work.

Run from project root:
    python src/validate_dataset.py
    python src/validate_dataset.py --data data/raw          # test against placeholder data
    python src/validate_dataset.py --data data --meta data/metadata.csv

Expected folder structure (what you'll build in India):
    data/
      pure/
        IMG_001.jpg
        ...
      adulterated/
        ratio_95_5/
        ratio_90_10/
        ratio_85_15/
        ratio_80_20/
        ratio_70_30/
        ratio_60_40/
        ratio_50_50/

Expected metadata CSV columns:
    file_name | ratio | lighting | distance | angle | lighting_method | illumination_color | notes

    lighting_method values: SVI_colored or ambient
    illumination_color values (SVI only): red, green, blue, yellow, white, mixed
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

EXPECTED_RATIO_FOLDERS = {
    "pure",
    "ratio_95_5",
    "ratio_90_10",
    "ratio_85_15",
    "ratio_80_20",
    "ratio_70_30",
    "ratio_60_40",
    "ratio_50_50",
}

VALID_LIGHTING_METHODS = {"SVI_colored", "ambient"}
REQUIRED_META_COLUMNS = {"file_name", "ratio", "lighting_method"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str):
    width = 62
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def ok(msg: str):
    print(f"  OK  {msg}")


def warn(msg: str):
    print(f"  WARN  {msg}")


def error(msg: str):
    print(f"  ERROR  {msg}")


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def scan_media(data_root: Path) -> list:
    """
    Walk data_root and return a list of dicts for every image/video found.
    Infers 'ratio' from the folder structure:
      pure/*           → ratio = "pure"
      adulterated/X/*  → ratio = X  (e.g. "ratio_90_10")
      adulterated/*    → ratio = "adulterated_unsorted"  (no sub-folder)
    """
    if not data_root.exists():
        error(f"Data root '{data_root}' does not exist.")
        sys.exit(1)

    found = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALL_MEDIA_EXTENSIONS:
            continue

        rel = path.relative_to(data_root)
        parts = rel.parts
        top = parts[0] if parts else "unknown"

        if top == "pure":
            ratio = "pure"
        elif top == "adulterated":
            if len(parts) >= 3:
                ratio = parts[1]           # sub-folder = ratio label
            else:
                ratio = "adulterated_unsorted"
        else:
            ratio = top                    # whatever the top folder is

        found.append({
            "file_name": path.name,
            "rel_path": str(rel),
            "ratio": ratio,
            "is_video": path.suffix.lower() in VIDEO_EXTENSIONS,
        })

    return found


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        return pd.read_csv(meta_path)
    except Exception as exc:
        warn(f"Could not read metadata file: {exc}")
        return None


# ---------------------------------------------------------------------------
# Audit sections
# ---------------------------------------------------------------------------

def audit_files(files: list, data_root: Path):
    section("1. FILES ON DISK")

    if not files:
        error(f"No image or video files found under '{data_root}'.")
        print("  Check that --data points to the right folder.")
        sys.exit(1)

    total = len(files)
    n_videos = sum(1 for f in files if f["is_video"])
    n_stills = total - n_videos
    print(f"  Total media files : {total}  ({n_stills} stills, {n_videos} videos)")

    # Group by ratio
    by_ratio: dict = defaultdict(list)
    for f in files:
        by_ratio[f["ratio"]].append(f)

    print(f"\n  Files per folder:")
    for ratio in sorted(by_ratio):
        count = len(by_ratio[ratio])
        unexpected = ratio not in EXPECTED_RATIO_FOLDERS
        flag = "  ** unexpected folder name **" if unexpected else ""
        print(f"    {ratio:<30} {count:>4}{flag}")

    unexpected_folders = [r for r in by_ratio if r not in EXPECTED_RATIO_FOLDERS]
    if unexpected_folders:
        warn(f"Unexpected folder names found: {unexpected_folders}")
        print(f"    Expected: {sorted(EXPECTED_RATIO_FOLDERS)}")

    missing_folders = [r for r in EXPECTED_RATIO_FOLDERS if r not in by_ratio]
    if missing_folders:
        warn(f"No files found for these expected ratios: {missing_folders}")

    return by_ratio


def audit_duplicates(by_ratio: dict):
    section("2. DUPLICATE FILENAMES")
    any_found = False
    for ratio, items in sorted(by_ratio.items()):
        names = [f["file_name"] for f in items]
        seen: set = set()
        dups = []
        for n in names:
            if n in seen:
                dups.append(n)
            seen.add(n)
        if dups:
            warn(f"Duplicate filenames in '{ratio}': {dups}")
            any_found = True
    if not any_found:
        ok("No duplicate filenames found.")


def audit_metadata(files: list, meta: pd.DataFrame, meta_path: Path):
    section("3. METADATA FILE")

    print(f"  Loaded {len(meta)} rows from {meta_path}")

    # Required column check
    missing_cols = REQUIRED_META_COLUMNS - set(meta.columns)
    if missing_cols:
        error(f"Required columns missing: {missing_cols}")
        print("  Skipping metadata cross-reference — fix column names first.")
        return
    ok(f"Required columns present: {sorted(REQUIRED_META_COLUMNS)}")

    # Cross-reference: filenames on disk vs metadata
    disk_names = {f["file_name"] for f in files}
    meta_names = set(meta["file_name"].dropna().astype(str))

    no_meta = disk_names - meta_names
    no_file = meta_names - disk_names

    section("3a. CROSS-REFERENCE: DISK vs METADATA")
    if no_meta:
        warn(f"{len(no_meta)} file(s) on disk have no metadata entry:")
        for name in sorted(no_meta):
            print(f"    {name}")
    else:
        ok("Every file on disk has a metadata entry.")

    if no_file:
        warn(f"{len(no_file)} metadata row(s) have no corresponding file:")
        for name in sorted(no_file):
            print(f"    {name}")
    else:
        ok("Every metadata entry points to an existing file.")

    # Lighting method breakdown
    if "lighting_method" in meta.columns:
        section("3b. LIGHTING METHOD BREAKDOWN")

        invalid = meta[~meta["lighting_method"].isin(VALID_LIGHTING_METHODS)]
        if not invalid.empty:
            warn(f"Unexpected lighting_method values in {len(invalid)} row(s):")
            print(invalid[["file_name", "lighting_method"]].to_string(index=False))
            print(f"    Valid values: {VALID_LIGHTING_METHODS}")

        try:
            breakdown = (
                meta.groupby(["ratio", "lighting_method"])
                .size()
                .unstack(fill_value=0)
            )
            print(f"\n  Shots per ratio × lighting method:")
            print(breakdown.to_string())
        except Exception:
            pass

        # Check every ratio has both SVI and ambient
        ratios_in_meta = sorted(meta["ratio"].dropna().unique())

        missing_svi = [
            r for r in ratios_in_meta
            if "SVI_colored" not in meta.loc[meta["ratio"] == r, "lighting_method"].values
        ]
        missing_ambient = [
            r for r in ratios_in_meta
            if "ambient" not in meta.loc[meta["ratio"] == r, "lighting_method"].values
        ]

        print()
        if missing_svi:
            warn(f"Ratios with no SVI_colored shots: {missing_svi}")
        else:
            ok("All ratios have SVI_colored shots.")

        if missing_ambient:
            warn(f"Ratios with no ambient shots: {missing_ambient}")
        else:
            ok("All ratios have ambient shots.")

    # Rows missing required values
    section("3c. MISSING VALUES IN METADATA")
    any_missing = False
    for col in sorted(REQUIRED_META_COLUMNS):
        if col not in meta.columns:
            continue
        n_null = meta[col].isna().sum()
        if n_null > 0:
            warn(f"Column '{col}' has {n_null} empty value(s).")
            any_missing = True
    if not any_missing:
        ok("No empty values in required columns.")


def audit_balance(by_ratio: dict):
    section("4. CLASS BALANCE SUMMARY")

    pure_count = len(by_ratio.get("pure", []))
    adult_count = sum(
        len(v) for k, v in by_ratio.items() if k != "pure"
    )
    total = pure_count + adult_count

    print(f"  Pure (label 0)        : {pure_count}")
    print(f"  Adulterated (label 1) : {adult_count}")
    print(f"  Total                 : {total}")

    if pure_count == 0 or adult_count == 0:
        error("One class has zero files — check your folder structure.")
    elif total < 50:
        warn(f"Only {total} total files. After augmentation, target 500+ training samples.")
    else:
        imbalance = abs(pure_count - adult_count) / max(pure_count, adult_count)
        if imbalance > 0.4:
            warn(
                f"Classes differ by {imbalance:.0%}. Consider collecting more images "
                f"from the smaller class before splitting."
            )
        else:
            ok(f"Class balance is acceptable ({imbalance:.0%} difference).")

    # Low-adulteration coverage (hardest to detect — must have enough)
    section("4a. LOW-ADULTERATION COVERAGE")
    print("  These ratios are hardest to detect and most important to have:")
    for r in ["ratio_95_5", "ratio_90_10", "ratio_85_15"]:
        count = len(by_ratio.get(r, []))
        flag = "  ** needs more images **" if count < 10 else ""
        print(f"    {r:<30} {count:>4} files{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit collected turmeric dataset before modeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python src/validate_dataset.py --data data --meta data/metadata.csv",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Path to data root folder (default: data/)",
    )
    parser.add_argument(
        "--meta",
        default="data/metadata.csv",
        help="Path to metadata CSV (default: data/metadata.csv)",
    )
    args = parser.parse_args()

    data_root = Path(args.data)
    meta_path = Path(args.meta)

    print(f"\nTurmeric Dataset Audit")
    print(f"  Data root : {data_root.resolve()}")
    print(f"  Metadata  : {meta_path.resolve()}")

    files = scan_media(data_root)
    by_ratio = audit_files(files, data_root)
    audit_duplicates(by_ratio)

    meta = load_metadata(meta_path)
    if meta is None:
        section("3. METADATA FILE")
        warn(f"No metadata file found at: {meta_path}")
        print("  Create a CSV with these columns and fill it as you shoot:")
        print("    file_name, ratio, lighting, distance, angle, lighting_method, illumination_color, notes")
        print("  Then re-run this script to check for gaps.")
    else:
        audit_metadata(files, meta, meta_path)

    audit_balance(by_ratio)

    section("AUDIT COMPLETE")
    print("  Fix ERRORs before moving to the split step.")
    print("  WARNINGs are worth reviewing but won't block training.")
    print()


if __name__ == "__main__":
    main()
