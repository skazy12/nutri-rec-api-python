import sys
from pathlib import Path

# raíz del proyecto = carpeta donde está modeloV01.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
