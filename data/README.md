# Data

This folder contains the experimental monitoring data collected during the LPBF builds.

## [Raw](raw)

The `raw` folder contains the original thermal and optical images as acquired continuously during each experiment, including process activity, recoater movements, and other captured events.

## [Curated](curated)

The `curated` folder contains images selected from the raw data for analysis:

* **In situ thermal:** Continuous thermal images acquired during laser processing, typically consisting of many images per layer.
* **Layer-wise thermal:** One thermal image per layer, selected at the same time step as the corresponding layer-wise optical image.
* **Layer-wise optical:** One optical image per layer, selected as the first obtainable quality optical image. Each image is time-matched to its corresponding layer-wise thermal image.


