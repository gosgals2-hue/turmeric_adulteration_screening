"""
normalize_new_pure.py — Repair and normalize a folder of newly transferred captures.

Transfer mangled the new pure-turmeric captures in three ways, all detectable from file
*content* rather than extension:

  1. Some ~1-minute VIDEOS were saved with a .HEIC extension (their container is QuickTime,
     signature "ftyp....qt  ", ~150-210 MB). These are real videos, not photos.
  2. Some .HEIC files are tiny 160x120 THUMBNAILS (< ~0.1 MP) — unusable junk.
  3. The real .MOV files are tiny (< 5 MB, some 0 bytes) Live-Photo/broken fragments, not the
     intentional 1-minute videos.

This script classifies each file by its actual bytes and:
  - renames QuickTime-in-.HEIC videos to .MOV (so frame extraction can use them);
  - converts genuine full-resolution HEIC photos to high-quality JPEG;
  - moves thumbnails and tiny/broken .MOV fragments into a _discarded/ subfolder (recoverable,
    never deleted).

Non-destructive: it renames, writes new .jpg files, and *moves* junk aside. Run --dry-run first.

REQUIREMENTS (for HEIC->JPG conversion only):  pip install pillow pillow-heif

USAGE (from project root):
    python src/normalize_new_pure.py --dry-run          # show the plan, change nothing
    python src/normalize_new_pure.py                    # apply
    python src/normalize_new_pure.py --src data/extra_new_pure --quality 95
"""
import argparse
import shutil
from pathlib import Path

MIN_PHOTO_MP = 1.0          # HEIC images below this are treated as thumbnails
MIN_REAL_MOV_BYTES = 5_000_000   # .MOV smaller than this is a Live-Photo/broken fragment
QT_BRANDS = {"qt", "qt  "}  # QuickTime ftyp brand (video mislabeled as .HEIC)


def ftyp_brand(path):
    """Return the major brand string from an ISO-BMFF file's ftyp box, or '' if unreadable."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        if head[4:8] != b"ftyp":
            return ""
        return head[8:12].decode("latin1").strip()
    except Exception:
        return ""


def heic_megapixels(path):
    """Lazily read a HEIC's dimensions (no full decode). Returns MP or None."""
    try:
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        with Image.open(path) as im:
            w, h = im.size
        return (w * h) / 1e6
    except Exception:
        return None


def classify(path):
    ext = path.suffix.lower()
    if ext == ".heic":
        if ftyp_brand(path) in QT_BRANDS:
            return "video_mislabeled"       # a QuickTime video wearing a .HEIC extension
        mp = heic_megapixels(path)
        if mp is None:
            return "unreadable_heic"
        return "thumbnail" if mp < MIN_PHOTO_MP else "photo"
    if ext == ".mov":
        return "mov_junk" if path.stat().st_size < MIN_REAL_MOV_BYTES else "video_ok"
    return "other"


def unique(path):
    """Avoid clobbering an existing target."""
    if not path.exists():
        return path
    i = 1
    while True:
        cand = path.with_name(f"{path.stem}__{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/extra_new_pure")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality for converted photos.")
    ap.add_argument("--discard-dir", default=None,
                    help="Where to move junk (default: <src>/_discarded).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"ERROR: {src} does not exist.")
    discard = Path(args.discard_dir) if args.discard_dir else src / "_discarded"

    files = [p for p in sorted(src.iterdir()) if p.is_file()]
    buckets = {}
    for p in files:
        buckets.setdefault(classify(p), []).append(p)

    print(f"Scanned {len(files)} files in {src}")
    for k in ("photo", "video_mislabeled", "thumbnail", "mov_junk", "video_ok",
              "unreadable_heic", "other"):
        if buckets.get(k):
            print(f"  {k:18} {len(buckets[k])}")
    if args.dry_run:
        print("\nDRY RUN — planned actions:")
        print(f"  rename {len(buckets.get('video_mislabeled', []))} mislabeled .HEIC -> .MOV")
        print(f"  convert {len(buckets.get('photo', []))} HEIC photos -> .jpg (q{args.quality})")
        print(f"  move {len(buckets.get('thumbnail', [])) + len(buckets.get('mov_junk', []))} "
              f"junk files -> {discard}")
        return

    # 1) rename mislabeled videos
    for p in buckets.get("video_mislabeled", []):
        dst = unique(p.with_suffix(".MOV"))
        p.rename(dst)
    # 2) convert real photos
    converted = failed = 0
    if buckets.get("photo"):
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        for i, p in enumerate(buckets["photo"], 1):
            try:
                with Image.open(p) as im:
                    im.convert("RGB").save(unique(p.with_suffix(".jpg")), "JPEG",
                                           quality=args.quality)
                converted += 1
            except Exception as e:
                print(f"  WARN could not convert {p.name}: {e}")
                failed += 1
            if i % 50 == 0:
                print(f"    converted {i}/{len(buckets['photo'])}")
    # 3) quarantine junk
    discard.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in buckets.get("thumbnail", []) + buckets.get("mov_junk", []):
        shutil.move(str(p), str(unique(discard / p.name)))
        moved += 1

    print(f"\nDone. renamed {len(buckets.get('video_mislabeled', []))} videos, "
          f"converted {converted} photos ({failed} failed), moved {moved} junk files to {discard}.")
    print("Originals (.HEIC) for converted photos are left in place; delete them once you've "
          "confirmed the .jpg files look right.")


if __name__ == "__main__":
    main()
