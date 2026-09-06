"""Can a kernel read its own previous output?

exp_010 needs thirty GPU hours and Kaggle's cap is twelve, so it has to run as
three sessions that each resume the last one's checkpoint. That plan rests
entirely on a kernel being allowed to list itself in `kernel_sources` and finding
the previous version's `/kaggle/working` mounted under `/kaggle/input`.

If it is not allowed, the resume is silently a fresh start: `restore_previous_run`
returns False, training begins again from exp_002, and three sessions produce one
session's progress. That is worth two minutes to establish now rather than at the
twelve-hour mark.

Run version 1 with no sources -- it writes a marker. Run version 2 with this
kernel listed as its own source -- it looks for the marker.
"""

from pathlib import Path

WORKING = Path("/kaggle/working")
MARKER = WORKING / "runs" / "polygon" / "weights" / "last.pt"


def main() -> None:
    print("=== /kaggle/input ===", flush=True)
    root = Path("/kaggle/input")
    if root.exists():
        for path in sorted(root.rglob("*"))[:40]:
            print(f"  {path}", flush=True)
    else:
        print("  (no /kaggle/input at all)", flush=True)

    # The exact glob exp_010 uses to find a previous session.
    found = sorted(root.rglob("runs/*/weights/last.pt")) if root.exists() else []
    print(f"\nexp_010's glob finds: {[str(p) for p in found]}", flush=True)
    print("VERDICT: " + ("self-reference works, resume is real"
                         if found else
                         "no previous output visible -- if this is version 2, "
                         "the resume plan does not work"), flush=True)

    # Leave the marker at the path exp_010 will look for, so version 2 is a
    # faithful test of the real glob rather than of a simpler one.
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_bytes(b"not a checkpoint, just a marker at the right path")
    print(f"\nwrote marker {MARKER}", flush=True)


if __name__ == "__main__":
    main()
