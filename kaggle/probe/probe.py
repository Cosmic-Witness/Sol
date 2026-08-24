"""Throwaway probe: report how the competition data is actually mounted.

Runs without a GPU so it costs no accelerator quota. Its only job is to remove
a guess from the main kernel's path configuration.
"""
from pathlib import Path

root = Path("/kaggle/input")
print("=== /kaggle/input top level ===")
for p in sorted(root.iterdir()):
    print(" ", p.name, "(dir)" if p.is_dir() else "(file)")

print("\n=== tree, depth 3 ===")
for p in sorted(root.rglob("*")):
    depth = len(p.relative_to(root).parts)
    if depth <= 3 and p.is_dir():
        n = sum(1 for _ in p.iterdir())
        print(f"  {'  ' * (depth - 1)}{p.relative_to(root)}/  ({n} entries)")

print("\n=== annotation json files ===")
for p in root.rglob("*.json"):
    print(" ", p, p.stat().st_size, "bytes")

print("\n=== image directories ===")
for name in ("train_images", "test_images"):
    for p in root.rglob(name):
        print(" ", p, sum(1 for _ in p.iterdir()), "files")
