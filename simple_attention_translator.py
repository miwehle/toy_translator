from pathlib import Path
import sys

TRANSLATOR_SRC_DIR = Path(__file__).resolve().parents[1] / "translator" / "src"
if str(TRANSLATOR_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSLATOR_SRC_DIR))

from src.train import main


if __name__ == "__main__":
    main()
