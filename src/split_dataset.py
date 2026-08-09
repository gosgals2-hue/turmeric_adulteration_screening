"""
split_dataset.py — PILE-AWARE split of the turmeric dataset into train/val/test.

Run from project root:
    python src/split_dataset.py                 # writes data/split/
    python src/split_dataset.py --dry-run       # print the plan, copy nothing
    python src/split_dataset.py --include-starch # also split the starch set

WHY pile-aware:
    A "pile" is a physical sample grouping (one mound photographed many times).
    If images from the SAME pile land in both train and test, the test score
    measures "recognize a known pile from a new angle", not "generalize to a new
    sample". The TEST set here is guaranteed pile-clean: no pile that appears in
    train appears in test.

Split design
    Pure: by default the held-out test pile is pure_E (backward compatible). As you add
        more pure piles you have three ways to route piles to TEST (the specificity fix):
          1. NAME a pile with "probe" in its id (e.g. pure_probe_store1) -> it is ALWAYS
             a held-out specificity probe: test-only, NEVER trained on. Use this for
             different-source turmeric (e.g. a US-store brand) you want as a clean
             "does it over-flag genuine turmeric?" test.
          2. --pure-test-piles pure_F,pure_G  -> force those exact piles to test.
          3. --pure-test-frac 0.25            -> deterministically send ~25% of the
             remaining (non-probe, non-explicit) pure piles to test, seeded.
        Anything not routed to test is a pure train pile. If none of the above selects a
        pure test pile, it falls back to pure_E so old behaviour is unchanged.
    Dye (2 piles per ratio, _d1/_d2): per ratio one day-pile -> train, the other
        -> test, ALTERNATING across ratios so both shooting days appear on both
        sides (no day/lighting confound).
    Validation: file-level carve from the TRAIN piles (both classes), seeded and
        per-pile stratified. Val is for model selection only; TEST is the only
        set whose numbers you report.
    Starch: excluded by default (SET ASIDE). With --include-starch it has one
        pile per ratio, so it can't be pile-split into test; placed in train only.

Quarantined images (metadata roi_quality == "segmentation_failure") are dropped
on load so they never enter any split.

Outputs:
    data/split/{train,val,test}/  + manifest.csv
"""

import sys
import shutil
import argparse
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

RANDOM_SEED = 42
VAL_FRACTION = 0.15
PURE_TEST_PILE = "pure_E"     # backward-compatible default held-out pure pile
PURE_PROBE_TAG = "probe"      # any pure pile whose id contains this is a test-only probe


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def load_metadata(meta_path: Path, data_root: Path, include_starch: bool) -> pd.DataFrame:
    df = pd.read_csv(meta_path, dtype=str).fillna("")
    required = {"file_name", "folder", "ratio", "pile_id"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: metadata is missing columns: {sorted(missing)}")

    df["rel_path"] = df["folder"].str.rstrip("/") + "/" + df["file_name"]
    df["source_path"] = df["rel_path"].map(lambda r: str((data_root / r).resolve()))

    def cls_of(pid):
        return 0 if pid.startswith("pure") else 1
    def type_of(pid):
        if pid.startswith("pure"):
            return "pure"
        if pid.startswith("yellow_dye"):
            return "yellow_dye"
        if pid.startswith("starch"):
            return "starch"
        return "unknown"
    def day_of(pid):
        for tag in ("_d1", "_d2"):
            if pid.endswith(tag):
                return tag[1:]
        return ""

    df["cls"] = df["pile_id"].map(cls_of)
    df["adul_type"] = df["pile_id"].map(type_of)
    df["day"] = df["pile_id"].map(day_of)

    if not include_starch:
        n_starch = (df["adul_type"] == "starch").sum()
        df = df[df["adul_type"] != "starch"].copy()
        print(f"  Excluded {n_starch} starch file(s) (SET ASIDE; use --include-starch to keep).")

    if "roi_quality" in df.columns:
        n_fail = (df["roi_quality"] == "segmentation_failure").sum()
        if n_fail:
            bad = df[df["roi_quality"] == "segmentation_failure"]["file_name"].tolist()
            df = df[df["roi_quality"] != "segmentation_failure"].copy()
            print(f"  Excluded {n_fail} quarantined segmentation_failure image(s): {bad}")

    return df


def assign_pile_splits(df: pd.DataFrame, pure_test_piles=None, pure_test_frac=0.0, seed=RANDOM_SEED):
    import random
    pile_split = {}
    piles_by_type = df.groupby("adul_type")["pile_id"].unique()

    # ---- pure piles: probes + explicit + optional fraction -> test; rest -> train ----
    pure_piles = sorted(piles_by_type.get("pure", []))
    explicit = set(pure_test_piles or []) & set(pure_piles)
    if pure_test_piles:
        unknown = set(pure_test_piles) - set(pure_piles)
        if unknown:
            print(f"  WARN: --pure-test-piles not found among pure piles (ignored): {sorted(unknown)}")
    probes = {p for p in pure_piles if PURE_PROBE_TAG in p}
    forced_test = probes | explicit
    remaining = [p for p in pure_piles if p not in forced_test]
    frac_test = set()
    if pure_test_frac and remaining:
        k = int(round(len(remaining) * pure_test_frac))
        if k > 0:
            frac_test = set(random.Random(seed).sample(remaining, min(k, len(remaining))))
    test_pure = forced_test | frac_test
    if not test_pure and PURE_TEST_PILE in pure_piles:      # backward-compatible fallback
        test_pure = {PURE_TEST_PILE}

    for p in pure_piles:
        if p in test_pure:
            reason = ("held-out pure specificity probe (never trained)"
                      if p in probes else "held-out pure test pile")
            pile_split[p] = ("test", reason)
        else:
            pile_split[p] = ("train", "pure train pile")
    if len(pure_piles) > 5 and len(test_pure) < 2:
        print(f"  HINT: {len(pure_piles)} pure piles but only {len(test_pure)} in test. "
              f"Specificity still rests on few piles — consider --pure-test-piles, "
              f"--pure-test-frac, or naming probe piles with '{PURE_PROBE_TAG}'.")
    print(f"  pure: {len(test_pure)} test pile(s) {sorted(test_pure)} "
          f"({len(probes)} probe), {len(pure_piles) - len(test_pure)} train.")

    dye = df[df["adul_type"] == "yellow_dye"]
    def ratio_key(r):
        try:
            return float(r.split("_")[0])
        except ValueError:
            return 0.0
    dye_ratios = sorted(dye["ratio"].unique(), key=ratio_key)
    for i, ratio in enumerate(dye_ratios):
        piles = sorted(dye[dye["ratio"] == ratio]["pile_id"].unique())
        d1 = next((p for p in piles if p.endswith("_d1")), None)
        d2 = next((p for p in piles if p.endswith("_d2")), None)
        if d1 and d2:
            if i % 2 == 0:
                pile_split[d1] = ("train", f"dye {ratio}: d1->train (alt even)")
                pile_split[d2] = ("test",  f"dye {ratio}: d2->test (alt even)")
            else:
                pile_split[d2] = ("train", f"dye {ratio}: d2->train (alt odd)")
                pile_split[d1] = ("test",  f"dye {ratio}: d1->test (alt odd)")
        else:
            for p in piles:
                pile_split[p] = ("train", f"dye {ratio}: single pile, train-only")
            print(f"  WARN: dye ratio {ratio} has one pile {piles} — no clean test side.")

    starch = df[df["adul_type"] == "starch"]
    if not starch.empty:
        print("  WARN: starch has one pile per ratio; assigning ALL to train.")
        for p in sorted(starch["pile_id"].unique()):
            pile_split[p] = ("train", "starch: single pile per ratio, train-only")

    return pile_split


def carve_val(df: pd.DataFrame, seed: int, val_fraction: float):
    for pid, grp in df[df["split"] == "train"].groupby("pile_id"):
        n = len(grp)
        k = int(round(n * val_fraction))
        if n >= 3:
            k = max(1, min(k, n - 1))
        else:
            k = 0
        if k == 0:
            continue
        chosen = grp.sample(n=k, random_state=seed)
        df.loc[chosen.index, "split"] = "val"
        df.loc[chosen.index, "split_reason"] = df.loc[chosen.index, "split_reason"] + " | file-level val carve"
    return df


def balance_train(df: pd.DataFrame, seed: int):
    """Downsample the majority class in TRAIN to a 50/50 pure:adulterated IMAGE count.

    Operates on train IMAGES only (videos feed frames separately). Excess majority-class
    images are moved to split 'unused_balance' — they are NOT deleted, just excluded from the
    balanced train split, so no collected data is lost on disk. Deterministic via seed.
    """
    vexts = tuple(VIDEO_EXTENSIONS)
    is_img = ~df["file_name"].str.lower().str.endswith(vexts)
    tr = df[(df["split"] == "train") & is_img].copy()
    tr["c"] = tr["pile_id"].map(lambda p: 0 if str(p).startswith("pure") else 1)
    counts = tr.groupby("c").size()
    if len(counts) < 2:
        print("  balance-train: only one class present in train — nothing to balance.")
        return df, 0
    target = int(counts.min())
    dropped = 0
    for c in counts.index:
        if int(counts[c]) > target:
            excess = tr[tr["c"] == c].sample(int(counts[c]) - target, random_state=seed)
            df.loc[excess.index, "split"] = "unused_balance"
            dropped += len(excess)
    print(f"  balance-train: train images balanced to {target}/{target} (pure/adulterated); "
          f"moved {dropped} majority-class images to 'unused_balance' (kept on disk).")
    return df, dropped


def copy_files(df: pd.DataFrame, out_root: Path):
    # Only IMAGES are copied into data/split/. Videos are large (multiple GB) and are read
    # in place from data/ via the manifest at frame-extraction time (see extract_frames.py),
    # so copying them here would needlessly duplicate gigabytes and is skipped.
    missing = copied = skipped_video = 0
    for _, row in df.iterrows():
        if row["split"] not in ("train", "val", "test"):
            continue   # e.g. 'unused_balance' rows are not part of the split
        if Path(row["file_name"]).suffix.lower() in VIDEO_EXTENSIONS:
            skipped_video += 1
            continue
        src = Path(row["source_path"])
        if not src.exists():
            missing += 1
            print(f"  WARN: source missing, skipped: {row['rel_path']}")
            continue
        dst = out_root / row["split"] / row["rel_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"  copied {copied} images; {skipped_video} videos left in place (pulled via manifest).")
    if missing:
        print(f"  WARN: {missing} file(s) referenced in metadata were not on disk.")


def print_summary(df: pd.DataFrame):
    section("SPLIT SUMMARY — by class")
    pivot = df.groupby(["cls", "split"]).size().unstack(fill_value=0)
    for col in ["train", "val", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["train", "val", "test"]]
    pivot.index = pivot.index.map({0: "pure(0)", 1: "adulterated(1)"})
    print(pivot.to_string())

    section("SPLIT SUMMARY — by pile")
    pv = df.groupby(["pile_id", "split"]).size().unstack(fill_value=0)
    for col in ["train", "val", "test"]:
        if col not in pv.columns:
            pv[col] = 0
    print(pv[["train", "val", "test"]].to_string())

    train_piles = set(df[df["split"] == "train"]["pile_id"])
    val_piles = set(df[df["split"] == "val"]["pile_id"])
    test_piles = set(df[df["split"] == "test"]["pile_id"])
    leaked = (train_piles | val_piles) & test_piles
    section("PILE-CLEANLINESS CHECK")
    if leaked:
        print(f"  FAIL: piles in both train/val and test: {sorted(leaked)}")
    else:
        print("  OK: no pile appears in both a training-side split and test.")
    print(f"  train piles: {len(train_piles)}  val piles (subset of train): "
          f"{len(val_piles)}  test piles: {len(test_piles)}")

    totals = df["split"].value_counts().to_dict()
    print(f"\n  Totals: train={totals.get('train',0)}  val={totals.get('val',0)}  "
          f"test={totals.get('test',0)}  total={len(df)}")
    print("\n  REMINDER: report metrics on TEST only. Val is for model selection.")


def main():
    ap = argparse.ArgumentParser(
        description="Pile-aware split of the turmeric dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", default="data")
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--out", default="data/split")
    ap.add_argument("--include-starch", action="store_true")
    ap.add_argument("--pure-test-piles", default="",
                    help="Comma-separated pure pile ids to force into TEST (e.g. pure_F,pure_G).")
    ap.add_argument("--pure-test-frac", type=float, default=0.0,
                    help="Deterministically route this fraction of remaining pure piles to TEST "
                         "(seeded). Probe/explicit piles are handled first.")
    ap.add_argument("--balance-train", action="store_true",
                    help="Downsample the majority class in TRAIN to a 50/50 pure:adulterated "
                         "image count (excess moved to 'unused_balance', not deleted).")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--freeze", action="store_true",
                    help="Mark the split FROZEN after writing (protects it from regeneration; use before final reporting).")
    args = ap.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)
    meta_path = Path(args.meta)

    print("\nTurmeric Dataset Split (pile-aware)")
    print(f"  Data     : {data_root.resolve()}")
    print(f"  Metadata : {meta_path.resolve()}")
    print(f"  Output   : {out_root.resolve()}")
    print(f"  Val carve: {args.val_fraction:.0%} of train files (seed={args.seed})")
    print(f"  Starch   : {'included (train-only)' if args.include_starch else 'excluded'}")
    if args.dry_run:
        print("  MODE     : DRY RUN (no files copied)")

    if not meta_path.exists():
        sys.exit(f"ERROR: metadata not found at '{meta_path}'.")

    # Provisional-vs-frozen model. The split is deterministic + leak-free, so while the
    # dataset is still growing it is safe to REGENERATE freely (provisional). Once you are
    # ready to report final metrics, run with --freeze; after that, regeneration is blocked
    # unless you explicitly --force (which you should almost never do).
    frozen_marker = out_root / "FROZEN.txt"
    is_frozen = frozen_marker.exists()
    if not args.dry_run and out_root.exists() and any(out_root.iterdir()):
        if is_frozen and not args.force:
            print(f"\n  ERROR: '{out_root}' is FROZEN (see FROZEN.txt).")
            print("  This is the split you report on; regenerating it would invalidate results.")
            print("  Pass --force ONLY if you truly intend to replace a frozen split.")
            sys.exit(1)
        why = "--force: replacing FROZEN split" if is_frozen else "regenerating PROVISIONAL split (deterministic, leak-free)"
        print(f"\n  {why}")
        shutil.rmtree(out_root)

    section("LOADING METADATA")
    df = load_metadata(meta_path, data_root, args.include_starch)
    print(f"  {len(df)} files across {df['pile_id'].nunique()} piles, "
          f"{df['ratio'].nunique()} ratios.")

    section("ASSIGNING PILES -> TRAIN / TEST")
    pure_test_piles = [p.strip() for p in args.pure_test_piles.split(",") if p.strip()]
    pile_split = assign_pile_splits(df, pure_test_piles=pure_test_piles,
                                    pure_test_frac=args.pure_test_frac, seed=args.seed)
    df["split"] = df["pile_id"].map(lambda p: pile_split.get(p, ("train", "unmapped"))[0])
    df["split_reason"] = df["pile_id"].map(lambda p: pile_split.get(p, ("train", "unmapped"))[1])

    section("CARVING VALIDATION FROM TRAIN PILES")
    df = carve_val(df, seed=args.seed, val_fraction=args.val_fraction)
    print(f"  Carved {int((df['split'] == 'val').sum())} files into val.")

    if args.balance_train:
        section("BALANCING TRAIN (50/50 pure:adulterated images)")
        df, _ = balance_train(df, seed=args.seed)

    if not args.dry_run:
        section("COPYING FILES")
        copy_files(df, out_root)
        print(f"  Files copied to: {out_root.resolve()}")

        section("SAVING MANIFEST")
        manifest_cols = ["file_name", "rel_path", "folder", "ratio",
                         "pile_id", "cls", "day", "split", "split_reason"]
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path = out_root / "manifest.csv"
        df[df["split"].isin(["train", "val", "test"])][manifest_cols].to_csv(manifest_path, index=False)
        print(f"  Saved: {manifest_path}")
        if args.freeze:
            import datetime
            frozen_marker.write_text(
                "FROZEN split - do not regenerate.\n"
                f"frozen_at: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"seed: {args.seed}  val_fraction: {args.val_fraction}\n"
                "Rationale: this is the split used for final reported metrics.\n")
            print("  ** SPLIT FROZEN ** (FROZEN.txt written) - regeneration now blocked without --force.")
        else:
            print("  Split is PROVISIONAL (regenerable). Run with --freeze before final reporting.")
    else:
        section("MANIFEST (dry run — not written)")
        print(df[["pile_id", "ratio", "cls", "split"]]
              .drop_duplicates("pile_id")
              .sort_values(["cls", "pile_id"])
              .to_string(index=False))

    print_summary(df)
    print()


if __name__ == "__main__":
    main()
