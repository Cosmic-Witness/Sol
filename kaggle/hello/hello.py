"""Smallest possible Kaggle kernel: proves the push -> run -> fetch-logs loop works.

No GPU, no data sources, no network. If this kernel's output comes back with the
line it prints, then code written in this repository can be executed on Kaggle
by an agent with only the CLI and an API token, and any later failure is a
problem with the experiment rather than with the plumbing.
"""

import platform
import sys
from datetime import datetime, timezone


def main() -> None:
    print("hello from a Kaggle kernel")
    print(f"utc         : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"platform    : {platform.platform()}")


if __name__ == "__main__":
    main()
