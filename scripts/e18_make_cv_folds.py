"""E18: build K-fold cross-validation splits over the full litho set.

The published in-distribution result rests on a single 60-image test split
(470 train / 58 val / 60 test out of 588). K-fold cross-validation lets every
image serve as held-out data exactly once, raising the in-distribution
evidence base from n=60 to n=588 and providing fold-to-fold variance for
metrics that are currently reported from a single training run.

Folds are materialised as directories of symlinks, so the existing
``scripts/baselines/train_baseline.py --data-root`` path works unchanged and
no image data is duplicated on disk.

Layout produced::

    <out>/fold0/images/{train,val,test}/*.png
    <out>/fold0/masks/{train,val,test}/*.png
    ...
    <out>/folds.json          assignment record, for reproducibility

Usage::

    python scripts/e18_make_cv_folds.py --k 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fraction of each fold's non-test remainder held out for validation. Matches
# the ratio of the original split (58 val / 528 non-test) closely enough that
# per-fold training budgets stay comparable to the published run.
VAL_FRACTION = 0.11


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="dataset/litho", help="source dataset root")
    p.add_argument("--out", default="dataset/litho_cv", help="output fold root")
    p.add_argument("--k", type=int, default=5, help="number of folds")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed")
    return p.parse_args()


def collect_pairs(src: Path) -> list[tuple[Path, Path]]:
    """Gather (image, mask) pairs across the source train/val/test splits.

    Pairing is by stem rather than by position so that a missing or extra file
    surfaces as an error here instead of as a silent misalignment during
    training.
    """
    pairs: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        img_dir, mask_dir = src / "images" / split, src / "masks" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.name.startswith("."):
                continue
            mask = mask_dir / img.name
            if not mask.exists():
                raise FileNotFoundError(f"image {img} has no mask at {mask}")
            pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs found under {src}")
    return pairs


def deterministic_shuffle(pairs: list[tuple[Path, Path]], seed: int) -> list[tuple[Path, Path]]:
    """Shuffle with a seeded RNG, after sorting by stem for a stable start."""
    ordered = sorted(pairs, key=lambda p: p[0].stem)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def link(target: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(target.resolve())


def main() -> None:
    args = parse_args()
    src = (REPO_ROOT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    out = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    pairs = deterministic_shuffle(collect_pairs(src), args.seed)
    n = len(pairs)
    if args.k < 2 or args.k > n:
        raise ValueError(f"--k must be in [2, {n}], got {args.k}")

    # Contiguous blocks over the shuffled order; the first n % k folds absorb
    # the remainder so fold sizes differ by at most one.
    base, extra = divmod(n, args.k)
    bounds, start = [], 0
    for i in range(args.k):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((start, stop))
        start = stop

    record: dict[str, object] = {
        "source": str(src.relative_to(REPO_ROOT)),
        "k": args.k,
        "seed": args.seed,
        "val_fraction": VAL_FRACTION,
        "total_pairs": n,
        "folds": {},
    }

    for i, (lo, hi) in enumerate(bounds):
        test = pairs[lo:hi]
        rest = pairs[:lo] + pairs[hi:]
        n_val = max(1, round(len(rest) * VAL_FRACTION))
        val, train = rest[:n_val], rest[n_val:]

        fold_dir = out / f"fold{i}"
        for split, items in (("train", train), ("val", val), ("test", test)):
            for img, mask in items:
                link(img, fold_dir / "images" / split / img.name)
                link(mask, fold_dir / "masks" / split / img.name)

        record["folds"][f"fold{i}"] = {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "test_stems": [img.stem for img, _ in test],
        }
        print(f"fold{i}: train={len(train)} val={len(val)} test={len(test)} -> {fold_dir}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "folds.json", "w") as f:
        json.dump(record, f, indent=2)

    covered = sum(len(v["test_stems"]) for v in record["folds"].values())
    distinct = len({s for v in record["folds"].values() for s in v["test_stems"]})
    print(f"\ntest coverage: {covered} assignments over {distinct} distinct images (total {n})")
    if not (covered == distinct == n):
        raise RuntimeError("fold test sets are not an exact partition of the dataset")
    print(f"assignment record -> {out/'folds.json'}")


if __name__ == "__main__":
    main()
