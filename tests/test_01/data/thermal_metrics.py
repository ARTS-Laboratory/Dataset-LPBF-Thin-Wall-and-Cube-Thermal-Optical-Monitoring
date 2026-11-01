"""Extract thermal statistics from single-frame NPZ files, without creating PNGs.

Run:
    python thermal_metrics.py
    python thermal_metrics.py --output "C:/path/to/results"

Dependencies: numpy, matplotlib. Each NPZ has 'temps' and 'time_unix_ns'.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def frame_suffix(path):
    match = re.search(r'(\d+)$', path.stem)
    return int(match.group(1)) if match else None


def load_metrics(path):
    with np.load(path, allow_pickle=False) as data:
        if 'temps' not in data.files or 'time_unix_ns' not in data.files:
            raise ValueError('NPZ must contain temps and time_unix_ns')
        a = np.asarray(data['temps'], dtype=np.float64)
        ts = np.asarray(data['time_unix_ns'])
        if a.ndim != 2 or not a.size or ts.size != 1:
            raise ValueError('Invalid temperature array or timestamp')
        timestamp_ns = int(ts.reshape(-1)[0])
        vals = a[np.isfinite(a)]
        if not vals.size:
            raise ValueError('No finite temperature values')
        median = float(np.median(vals))
        # Relative hot-area metric works without a known absolute print temperature.
        hot_fraction = float(np.mean(vals >= median + 5.0) * 100.0)
        return dict(filename=path.name, original_frame_id=frame_suffix(path),
                    timestamp_unix_ns=timestamp_ns,
                    timestamp_utc=datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat(),
                    min_c=float(vals.min()), max_c=float(vals.max()),
                    mean_c=float(vals.mean()), std_c=float(vals.std(ddof=0)),
                    median_c=median, p95_c=float(np.percentile(vals, 95)),
                    p99_c=float(np.percentile(vals, 99)),
                    range_c=float(vals.max()-vals.min()),
                    hot_pixels_pct=hot_fraction,
                    valid_pixel_count=int(vals.size),
                    invalid_pixel_count=int(a.size-vals.size))


def save_plot(rows, out, xkey, xlabel, suffix):
    x = np.array([r[xkey] for r in rows], dtype=float)
    traces = [
        ('Minimum', 'min_c'), ('Maximum', 'max_c'),
        ('Average', 'mean_c'), ('Standard deviation', 'std_c'),
        ('Median', 'median_c'), ('95th percentile', 'p95_c'),
        ('99th percentile', 'p99_c'), ('Temperature range', 'range_c')
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    for ax, (label, key) in zip(axes.flat, traces):
        ax.plot(x, [r[key] for r in rows], marker='o', linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Temperature (°C)' if key != 'std_c' else 'Temperature std. dev. (°C)')
        ax.grid(alpha=0.3)
        if len(x) == 1: ax.set_xlim(x[0]-0.5, x[0]+0.5)
    fig.suptitle('Thermal frame statistics vs ' + suffix, fontsize=15)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_trigger_plot(rows, out):
    x = [r['elapsed_s'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for ax, key, title, unit in [
        (axes[0], 'p99_c', '99th-percentile temperature', '°C'),
        (axes[1], 'hot_pixels_pct', 'Pixels >= frame median + 5 °C', '%'),
        (axes[2], 'delta_p99_c', 'Change in 99th-percentile temperature', '°C')
    ]:
        ax.plot(x, [r[key] if r[key] != '' else np.nan for r in rows], '-o')
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Elapsed timestamp time (s)')
    fig.suptitle('Candidate thermal trigger features (exploratory only)')
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_folder', nargs='?', type=Path, default=Path(__file__).resolve().parent / 'raw')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    source = args.input_folder.expanduser().resolve()
    # Default directory structure: script_folder/raw -> script_folder/processed.
    # An explicit input folder also writes to its sibling processed folder.
    out = (args.output if args.output else source.parent / 'processed'/ 'thermal analysis').expanduser().resolve()
    if not source.is_dir():
        parser.error(f'Input raw folder does not exist: {source}')
    paths = list(source.glob('*.npz'))
    if not paths: parser.error(f'No NPZ files in {source}')
    rows = []
    for path in paths:
        try:
            rows.append(load_metrics(path))
        except Exception as exc:
            print(f'SKIPPING {path.name}: {exc}')
    if not rows: parser.error('No valid NPZ recordings')
    # Always order by the actual recorded timestamp rather than filename.
    rows.sort(key=lambda r: (r['timestamp_unix_ns'], r['filename']))
    first = rows[0]['timestamp_unix_ns']
    for i, row in enumerate(rows):
        row['sequence_number'] = i + 1
        # Integer nanosecond subtraction avoids precision loss near Unix epoch.
        row['elapsed_s'] = (row['timestamp_unix_ns'] - first) / 1e9
        row['interval_from_previous_s'] = '' if i == 0 else (row['timestamp_unix_ns'] - rows[i-1]['timestamp_unix_ns']) / 1e9
        row['delta_max_c'] = '' if i == 0 else row['max_c'] - rows[i-1]['max_c']
        row['delta_mean_c'] = '' if i == 0 else row['mean_c'] - rows[i-1]['mean_c']
        row['delta_p99_c'] = '' if i == 0 else row['p99_c'] - rows[i-1]['p99_c']
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'thermal_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, out / 'thermal_statistics_vs_time.png', 'elapsed_s', 'Elapsed timestamp time (s)', 'elapsed time')
    save_plot(rows, out / 'thermal_statistics_vs_frame.png', 'sequence_number', 'Analyzed frame number', 'frame number')
    save_trigger_plot(rows, out / 'thermal_trigger_features.png')
    print(f'Processed {len(rows)} thermal NPZ frames. Saved results to: {out}')
    for r in rows:
        print(f"{r['filename']:24s} frame {r['sequence_number']:4}  t={r['elapsed_s']:8.3f}s  "
              f"min={r['min_c']:.2f} max={r['max_c']:.2f} mean={r['mean_c']:.3f} std={r['std_c']:.3f} °C")


if __name__ == '__main__':
    main()
