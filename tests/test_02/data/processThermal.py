""" thermal NPZ converter. Reads .npz from raw folder and writes PNGs to ./processed/thermal images folder.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

def clear_terminal():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def gimmeFileNames(path, includeFolders=False):
    folder = Path(path)
    if includeFolders:
        return sorted(item.name for item in folder.iterdir())
    return sorted(file.name for file in folder.iterdir() if file.is_file())


def buildFilePaths(path):
    return [str(Path(path) / name) for name in gimmeFileNames(path)]


def _scalar_from_npz(data, key, default=None):
    if key not in data.files:
        return default
    values = np.asarray(data[key]).reshape(-1)
    if values.size == 0:
        return default
    return values[0].item()


def loadRecording(path):
    """Load a thermal recording from an NPZ file. single-frame file has no within-file capture interval.
    """
    path = Path(path)
    print(f"Loading {path.name}.")

    with np.load(path, allow_pickle=False) as data:
        save_time_unix_ns = None
        if "temps" in data.files:
            temps = np.asarray(data["temps"], dtype=np.float32)
            if temps.ndim != 2 or 0 in temps.shape:
                raise ValueError(f"Expected temps with shape [height, width]; got {temps.shape}.")
            frames = temps[np.newaxis, ...]
            recording_format = "single_frame"
            if "time_unix_ns" in data.files:
                stamp = np.asarray(data["time_unix_ns"])
                if stamp.size != 1 or not np.issubdtype(stamp.dtype, np.integer):
                    raise ValueError("time_unix_ns must be one integer Unix timestamp.")
                save_time_unix_ns = int(stamp.reshape(-1)[0])
            elapsed_s = np.array([0.0], dtype=np.float64)
            frame_timestamps_ns = np.array([], dtype=np.int64)
            capture_interval_s = None
            timestamp = "" if save_time_unix_ns is None else _format_save_time(save_time_unix_ns)
        elif "frames" in data.files:
            frames = np.asarray(data["frames"], dtype=np.float32)
            recording_format = "multi_frame"
        else:
            raise KeyError(
                f"{path.name} contains neither 'temps' (new server) nor 'frames' (legacy). "
                f"Available keys: {data.files}"
            )
        if frames.ndim != 3:
            raise ValueError(
                f"Expected frames with shape [N, height, width]; got {frames.shape}."
            )
        if frames.shape[0] == 0:
            raise ValueError(f"{path.name} contains no thermal frames.")

        if recording_format == "multi_frame":
            timestamp = _scalar_from_npz(data, "timestamp", "")
            capture_interval_s = _scalar_from_npz(data, "capture_interval_s", None)

        if recording_format == "multi_frame" and "frame_timestamps_ns" in data.files:
            frame_timestamps_ns = np.asarray(
                data["frame_timestamps_ns"], dtype=np.int64
            ).reshape(-1)
        elif recording_format == "multi_frame":
            frame_timestamps_ns = np.array([], dtype=np.int64)

        if recording_format == "single_frame":
            pass  # One shot; elapsed time within this file is zero.
        elif "elapsed_s" in data.files:
            elapsed_s = np.asarray(data["elapsed_s"], dtype=np.float64).reshape(-1)
        elif frame_timestamps_ns.size:
            elapsed_s = (
                frame_timestamps_ns - frame_timestamps_ns[0]
            ).astype(np.float64) / 1e9
        elif capture_interval_s is not None:
            elapsed_s = (
                np.arange(frames.shape[0], dtype=np.float64)
                * float(capture_interval_s)
            )
        else:
            raise KeyError(
                "The NPZ has no elapsed_s, frame_timestamps_ns, or "
                "capture_interval_s timing information."
            )

    if elapsed_s.size != frames.shape[0]:
        raise ValueError(
            f"Frame/time count mismatch: {frames.shape[0]} frames and "
            f"{elapsed_s.size} elapsed times."
        )
    if frame_timestamps_ns.size not in (0, frames.shape[0]):
        raise ValueError(
            f"Frame/timestamp count mismatch: {frames.shape[0]} frames and "
            f"{frame_timestamps_ns.size} timestamps."
        )
    if not np.all(np.isfinite(elapsed_s)) or np.any(np.diff(elapsed_s) < 0):
        raise ValueError("Elapsed capture times must be finite and nondecreasing.")

    elapsed_s = elapsed_s - elapsed_s[0]
    print(
        f"Loaded {frames.shape[0]} frames covering "
        f"{elapsed_s[-1]:.3f} seconds."
    )

    return {
        "frames": frames,
        "timestamp": timestamp,
        "frame_timestamps_ns": frame_timestamps_ns,
        "elapsed_s": elapsed_s,
        "capture_interval_s": (
            None if capture_interval_s is None else float(capture_interval_s)
        ),
        "recording_format": recording_format,
        "save_time_unix_ns": save_time_unix_ns,
    }


def _format_save_time(timestamp_ns):
    """Preserve integer nanoseconds in a UTC ISO timestamp."""
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{nanoseconds:09d}Z"


def loadData(path, name=None):
    """Backward-compatible loader used by older scripts."""
    recording = loadRecording(path)
    return (
        recording["frames"],
        recording["timestamp"],
        recording["frame_timestamps_ns"],
    )


def selectEqualIntervalFrames(elapsed_s, interval_s=None):
    """
    Select the recorded frame nearest to each equal target time.
    interval_s=None exports every stored frame. This is the preferred setting
    for NPZ files produced by the interval-enabled acquisition code because
    those frames were already captured at the requested equal interval.
    """
    elapsed_s = np.asarray(elapsed_s, dtype=np.float64)
    frame_count = elapsed_s.size
    if frame_count == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64)

    if interval_s is None:
        return np.arange(frame_count, dtype=int), elapsed_s.copy()

    interval_s = float(interval_s)
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("EXPORT_INTERVAL_S must be greater than zero.")

    positive_gaps = np.diff(elapsed_s)
    positive_gaps = positive_gaps[positive_gaps > 0]
    if positive_gaps.size:
        source_interval_s = float(np.median(positive_gaps))
        if interval_s < source_interval_s * 0.95:
            raise ValueError(
                f"Requested export interval {interval_s:g} s is shorter than "
                f"the recorded interval (approximately {source_interval_s:g} s). "
                "Missing intermediate thermal frames cannot be reconstructed."
            )

    target_count = int(np.floor(elapsed_s[-1] / interval_s)) + 1
    target_times_s = np.arange(target_count, dtype=np.float64) * interval_s

    right = np.searchsorted(elapsed_s, target_times_s, side="left")
    right = np.clip(right, 0, frame_count - 1)
    left = np.clip(right - 1, 0, frame_count - 1)

    choose_left = np.abs(elapsed_s[left] - target_times_s) <= np.abs(
        elapsed_s[right] - target_times_s
    )
    indices = np.where(choose_left, left, right).astype(int)

    # If irregular source timing maps two targets to one physical frame, write
    # that physical frame only once.
    keep = np.concatenate(([True], np.diff(indices) != 0))
    return indices[keep], target_times_s[keep]


def _write_inferno_png(image_8bit, image_path):
    """Write an Inferno-colored PNG with OpenCV or a portable fallback."""
    if cv2 is not None:
        color = cv2.applyColorMap(image_8bit, cv2.COLORMAP_INFERNO)
        if not cv2.imwrite(str(image_path), color):
            raise OSError(f"OpenCV could not write {image_path}.")
        return

    from matplotlib import colormaps
    from PIL import Image

    rgb = colormaps["inferno"](image_8bit / 255.0, bytes=True)[..., :3]
    Image.fromarray(rgb, mode="RGB").save(image_path, format="PNG")


def saveImages(
    path,
    output,
    interval_s=None,
    temp_min_c=None,
    temp_max_c=None,
):
    """Export timestamped thermal PNG images"""
    started = time.perf_counter()
    path = Path(path)
    recording = loadRecording(path)
    frames = recording["frames"]
    elapsed_s = recording["elapsed_s"]

    if recording["recording_format"] == "single_frame" and interval_s is not None:
        raise ValueError(
            "Use EXPORT_INTERVAL_S = None for one-frame-per-file NPZs. "
            "Capture timing is controlled by LabVIEW; this exporter processes each file."
        )

    indices, target_times_s = selectEqualIntervalFrames(elapsed_s, interval_s)
    selected = frames[indices]

    finite = frames[np.isfinite(frames)]
    if finite.size == 0 or np.any(~np.any(np.isfinite(frames), axis=(1, 2))):
        raise ValueError("Every thermal frame must contain at least one finite temperature.")
    low_c = float(finite.min()) if temp_min_c is None else float(temp_min_c)
    high_c = float(finite.max()) if temp_max_c is None else float(temp_max_c)
    if low_c == high_c and temp_min_c is None and temp_max_c is None:
        low_c -= 0.5
        high_c += 0.5
    if not np.isfinite(low_c) or not np.isfinite(high_c) or high_c <= low_c:
        raise ValueError(
            f"Invalid temperature scale: minimum={low_c}, maximum={high_c}."
        )

    # Save single-frame PNGs directly in thermal images, without per-file subfolders.
    # Keep legacy multi-frame recordings in separate folders to avoid naming collisions.
    folder = Path(output) if recording["recording_format"] == "single_frame" else Path(output) / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    for image_number, (frame_index, target_s, frame) in enumerate(
        zip(indices, target_times_s, selected), start=1
    ):
        actual_s = float(elapsed_s[frame_index])
        normalized = np.clip((frame - low_c) / (high_c - low_c), 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        image_8bit = np.rint(normalized * 255.0).astype(np.uint8)
        if recording["recording_format"] == "single_frame":
            filename = f"{path.stem}.png"
        else:
            filename = (
                f"thermal_{image_number:06d}_"
                f"target_{target_s:010.3f}s_actual_{actual_s:010.3f}s.png"
            )
        image_path = folder / filename
        _write_inferno_png(image_8bit, image_path)

    hottest_local_index = int(np.argmax(np.max(np.where(np.isfinite(selected), selected, -np.inf), axis=(1, 2))))
    hottest_source_index = int(indices[hottest_local_index])
    elapsed = time.perf_counter() - started

    print(
        f"Saved {len(indices)} PNG images to {folder} "
        f"in {elapsed:.2f} seconds."
    )
    print(
        f"Temperature color scale: {low_c:.3f} to {high_c:.3f} °C. "
        f"Hottest exported source frame: {hottest_source_index}.\n"
    )

    return {
        "folder": folder,
        "exported_indices": indices,
        "target_times_s": target_times_s,
        "hottest_source_index": hottest_source_index,
    }


# ================================================================
# MAIN PROCESSING APPLICATION
# ================================================================

"""Convert thermal .npz files from ./raw to ./processed.

"""

import argparse
import json

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Always use the raw folder beside this script when run without arguments.
INPUT_PATH = BASE_DIR / "raw"
SERVER_STATE_PATH = BASE_DIR.parent / "last_recording.json"

# Store converted PNG images in processed/thermal images.
OUTPUT_IMAGE_PATH = BASE_DIR / "processed"


# one PNG per NPZ. Timing is controlled by LabVIEW, not this exporter.
# A larger interval is supported only for legacy multi-frame 'frames' NPZs;
EXPORT_INTERVAL_S = None

# None/None uses a min/max scale per NPZ (per image for thermal server).
# For identical colors across different tests, set fixed limits such as
# TEMPERATURE_MIN_C = 20.0 and TEMPERATURE_MAX_C = 120.0.
TEMPERATURE_MIN_C = None
TEMPERATURE_MAX_C = None



def read_server_input_folder():
    """Read shared location data without importing or running the camera server."""
    try:
        state = json.loads(SERVER_STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"No server folder information was found at {SERVER_STATE_PATH}. "
            "Run the updated thermalServer.py and save at least one thermal NPZ first. "
            "Alternatively, pass a folder argument or use --choose-folder."
        ) from error
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot read server folder information at {SERVER_STATE_PATH}: {error}. "
            "Save another frame with the updated server or pass a folder explicitly."
        ) from error
    folder = state.get("input_folder") if isinstance(state, dict) else None
    if not isinstance(folder, str) or not folder.strip():
        raise RuntimeError(f"Missing valid input_folder in {SERVER_STATE_PATH}.")
    path = Path(folder)
    if not path.is_absolute():
        raise RuntimeError(f"Expected an absolute folder path in {SERVER_STATE_PATH}.")
    print(f"Using recording folder reported by thermalServer.py: {path}")
    return path


def choose_input_folder():
    """Ask for the actual recording folder; do not assume a script-relative path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError(
            'Folder selection is unavailable. Pass the LabVIEW folder explicitly, '
            'for example: python npzProcessing.py "C:\\CameraData\\Test_019\\raw"'
        ) from error

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(
            parent=root,
            title="Select the folder entered in LabVIEW Output Folder",
            initialdir=str(BASE_DIR),
            mustexist=True,
        )
    except tk.TclError as error:
        raise RuntimeError(
            'Cannot open the folder chooser. Pass the LabVIEW folder explicitly, '
            'for example: python npzProcessing.py "C:\\CameraData\\Test_019\\raw"'
        ) from error
    finally:
        if root is not None:
            root.destroy()
    return Path(selected) if selected else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert thermal NPZ files from raw to processed/thermal images."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Optional input folder; defaults to raw beside this Python script.",
    )
    parser.add_argument(
        "--choose-folder", action="store_true",
        help="Choose a folder manually instead of using the last server folder.",
    )
    args = parser.parse_args(argv)
    if args.folder is not None and args.choose_folder:
        parser.error("Use either a folder argument or --choose-folder, not both.")
    # Keep terminal logs visible.

    try:
        if args.choose_folder:
            input_path = choose_input_folder()
            if input_path is None:
                print("No folder selected. Nothing was processed.")
                return 0
        elif args.folder is not None:
            input_path = args.folder
        elif INPUT_PATH is not None:
            input_path = INPUT_PATH
        else:
            input_path = read_server_input_folder()
    except RuntimeError as error:
        parser.error(str(error))

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_dir():
        parser.error(f"Input folder does not exist or is not a directory: {input_path}")

    # Read completed NPZ files directly in the selected folder. Atomic-save
    # temporary files such as Thermal_0001.npz.tmp are excluded by this pattern.
    npz_paths = sorted(path for path in input_path.glob("*.npz") if path.is_file())
    if not npz_paths:
        parser.error(
            f"No .npz recordings were found directly in {input_path}. "
            "Select the exact folder containing the saved thermal files."
        )

    OUTPUT_IMAGE_PATH.mkdir(parents=True, exist_ok=True)

    print(f"NPZ input folder: {input_path}")
    print(f"Found {len(npz_paths)} NPZ recording(s).")
    print(f"PNG output directory: {OUTPUT_IMAGE_PATH}")
    print()

    successful = 0
    failed = 0
    for path in npz_paths:
        print("-" * 70)
        try:
            saveImages(
            path=path,
            output=OUTPUT_IMAGE_PATH,
            interval_s=EXPORT_INTERVAL_S,
            temp_min_c=TEMPERATURE_MIN_C,
            temp_max_c=TEMPERATURE_MAX_C,
            )
            successful += 1
        except Exception as exc:
            failed += 1
            print(f"ERROR processing {path.name}: {exc}")

    print(f"Completed: {successful} NPZ files converted successfully, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
