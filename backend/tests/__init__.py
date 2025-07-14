import sys
from pathlib import Path
print(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
