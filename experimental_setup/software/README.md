# Record

LabVIEW interface for operating Optris thermal cameras and recording thermal image data.

## Requirements

Before using the software:

1. Install the Optris OTC SDK.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Python Dependencies

The following Python packages are required:

* `numpy`
* `opencv-python`

> **Note:** The Optris OTC SDK is not installed through `pip` and must be installed separately.

## Current Features

* Live thermal image streaming to LabVIEW
* Thermal recording to compressed `.npz` files
* Camera focus control
* Flag event triggering
* Thermal video streaming over a dedicated socket connection

## Usage

1. Open "thermal_optical_camera.vi".
2. Create a destination folder for the optical images for each time you run the VI. (Folder1, Folder2, Folder3, etc...)
3. Enter "1" for capture intervals for both thermal and optical camera.
4. Run the VI. Optical camera will begin capturing.
5. Wait for the terminal to indicate that camera calibration has completed and temperature measurements are reliable.
6. Press "Record" button on front panel to begin recording thermal data.
7. Press "Stop Recording" button to save thermal recording.
8. Stop the VI. Optical camera will stop writing files.
9. Open "...\software\Recordings" folder to verify npz file has been written.
10. Open the python workspace and run npzProcessing.py. Images will written to "...\software\npz Processor\Images" folder.

## Recording Format

Recordings are saved as compressed `.npz` files containing:

* `frames` — thermal image data stored as `float32` temperature values.
* `frame_timestamps_ns` — per-frame timestamps.
* `timestamp` — recording save time.

## Notes

> For best temperature accuracy, trigger a **Flag** event before recording.

Camera preview and recording are controlled through a Python server running in the background and communicating with LabVIEW through TCP socket connections.
