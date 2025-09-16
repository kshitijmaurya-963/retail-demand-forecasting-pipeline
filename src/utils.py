import os
from datetime import datetime

def timestamped(path_dir, stem):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(path_dir, f"{stem}_{ts}.csv")
