"""Convert thermal .npz files from ./raw to ./processed/thermal images.

Keep functionFiles.py beside this script. Optional command-line folder selection
is supported, but running without arguments reads the project raw folder.
"""

import argparse
import json
from pathlib import Path

import functionFiles as thermal


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Always use the raw folder beside this script when run without arguments.
INPUT_PATH = BASE_DIR / "raw"
SERVER_STATE_PATH = BASE_DIR.parent / "last_recording.json"

# Store converted PNG images in processed/thermal images.
OUTPUT_IMAGE_PATH = BASE_DIR / "processed" / "thermal images"
# Preserve the existing CSV export capability in processed/thermal images/CSV.
#OUTPUT_CSV_PATH = BASE_DIR / "processed"/ "thermal images" / "CSV"

# one PNG per NPZ. Timing is controlled by LabVIEW, not this exporter.
# A larger interval is supported only for legacy multi-frame 'frames' NPZs;
EXPORT_INTERVAL_S = None

# None/None uses a min/max scale per NPZ (per image for Austin's server).
# For identical colors across different tests, set fixed limits such as
# TEMPERATURE_MIN_C = 20.0 and TEMPERATURE_MAX_C = 120.0.
TEMPERATURE_MIN_C = None
TEMPERATURE_MAX_C = None

# It exports the hottest selected frame.
SAVE_HOTTEST_FRAME_CSV = True


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
    thermal.clear_terminal()

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
    if SAVE_HOTTEST_FRAME_CSV:
        OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)

    print(f"NPZ input folder: {input_path}")
    print(f"Found {len(npz_paths)} NPZ recording(s).")
    print(f"PNG output directory: {OUTPUT_IMAGE_PATH}")
    if SAVE_HOTTEST_FRAME_CSV:
        print(f"CSV output directory: {OUTPUT_CSV_PATH}")
    print()

    for path in npz_paths:
        print("-" * 70)
        result = thermal.saveImages(
            path=path,
            output=OUTPUT_IMAGE_PATH,
            interval_s=EXPORT_INTERVAL_S,
            temp_min_c=TEMPERATURE_MIN_C,
            temp_max_c=TEMPERATURE_MAX_C,
        )

        if SAVE_HOTTEST_FRAME_CSV:
            thermal.saveCSV(
                path=path,
                name=path.name,
                output=OUTPUT_CSV_PATH,
                frameIndex=result["hottest_source_index"],
            )

    print("All recordings processed and images created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
