#!/usr/bin/env python3
import shutil
from pathlib import Path


def main() -> None:
    output_dir = Path("output-preds")
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print("Cleared output-preds")


if __name__ == "__main__":
    main()
