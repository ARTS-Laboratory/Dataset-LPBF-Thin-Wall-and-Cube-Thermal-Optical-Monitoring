# Data

Contains raw, processed, and curated thermal and optical monitoring data, along with scripts for data processing.

## Raw

- Contains original thermal (`.npz`) and optical (`.bin`) data collected during the experiment. 
- Run 'processOptical.py' to create .png files from  .bin files captured by optical camera. 
- Run 'processThermal.py' to create .png files from .npz files captured by thermal camera. 

## Processed

Contains PNG images generated from the raw thermal and optical data.

## Curated

Contains selected PNG images from the processed dataset for analysis, visualization, and machine learning.