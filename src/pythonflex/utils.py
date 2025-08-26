import os
import re  # For sanitization (built-in, minimal regex)
import tempfile
import joblib
import numpy as np
import pandas as pd

# Constants
TMP_ROOT = ".tmp"
VALID_EXTS = {".feather", ".npy", ".pkl"}

# Helper to sanitize names (make filesystem-safe)
def _sanitize(name):
    if not name:
        return "data"
    # Replace forbidden/problematic chars with '_', collapse multiples, strip edges
    safe = re.sub(r'[<>:"/\\|?*$,\s]+', '_', str(name).strip())
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe if safe else "data"

# Helper to get safe path
def _safe_path(category, name=None, ext=".pkl"):
    safe_category = _sanitize(category)
    dir_path = os.path.join(TMP_ROOT, safe_category)
    os.makedirs(dir_path, exist_ok=True)
    safe_name = _sanitize(name) if name else "data"
    return os.path.join(dir_path, f"{safe_name}{ext}")

# Save function
def dsave(data, category, name=None, path=None):  # 'path' ignored for compatibility with old code
    # If data is dict and no name, recurse on each item
    if name is None and isinstance(data, dict):
        for k, v in data.items():
            dsave(v, category, k)
        return

    # Choose best extension based on type
    if isinstance(data, pd.DataFrame):
        ext = ".feather"
        save_func = lambda p: data.to_feather(p)
    elif isinstance(data, np.ndarray):
        ext = ".npy"
        save_func = lambda p: np.save(p, data, allow_pickle=False)
    else:
        ext = ".pkl"
        save_func = lambda p: joblib.dump(data, p, compress=0)  # Add compress=3 if needed

    target = _safe_path(category, name, ext)

    # Atomic save: Write to temp file, then rename
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(target), delete=False) as tf:
        tmp_path = tf.name
        tf.close()  # Close so save_func can write
        save_func(tmp_path)
    os.replace(tmp_path, target)  # Atomic move

# Load function
def dload(category, name=None, path=None):  # 'path' ignored for compatibility
    dir_path = os.path.join(TMP_ROOT, _sanitize(category))

    if not os.path.exists(dir_path):
        return {}

    if name is None:
        # Load all in category as dict
        out = {}
        for filename in os.listdir(dir_path):
            if not any(filename.endswith(ext) for ext in VALID_EXTS):
                continue
            k = os.path.splitext(filename)[0]  # Key from filename (without ext)
            full_path = os.path.join(dir_path, filename)
            try:
                if filename.endswith(".feather"):
                    out[k] = pd.read_feather(full_path)
                elif filename.endswith(".npy"):
                    out[k] = np.load(full_path, mmap_mode="r")  # MMap for perf
                elif filename.endswith(".pkl"):
                    out[k] = joblib.load(full_path, mmap_mode="r")  # MMap for perf
            except (EOFError, ValueError, OSError):
                print(f"Warning: '{full_path}' is corrupted. Skipping...")
                os.remove(full_path)  # Delete corrupted file
        return out

    # Load specific name (try extensions in order)
    for ext in VALID_EXTS:
        target = _safe_path(category, name, ext)
        if os.path.exists(target):
            try:
                if ext == ".feather":
                    return pd.read_feather(target)
                elif ext == ".npy":
                    return np.load(target, mmap_mode="r")  # MMap for perf
                elif ext == ".pkl":
                    return joblib.load(target, mmap_mode="r")  # MMap for perf
            except (EOFError, ValueError, OSError):
                print(f"Warning: '{target}' is corrupted. Deleting and returning {{}}...")
                os.remove(target)
                return {}
    return {}

