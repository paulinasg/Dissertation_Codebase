# Repository Structure Documentation

## Root Directories

### `code/`
Contains scripts and programs for processing, manipulating, and analysing the dataset.

- **`4d_label/`**: Scripts for 4D model labeling and coloring of the ground truth 4D-DRESS files
  - Contains custom and default color models
  - Includes label fixes and adjustments

- **`align/`**: Alignment tools and scripts for model alignment between 4D-DRESS ground truths and PIFuHD results

- **`compare/`**: Comparison utilities for analysing accuracies of models using different methods

- **`convert/`**: Conversion between .pkl compressed, .txt, and .obj model formats

- **`pifuhd_label/`**: Labeling scripts for segmentation transfer onto PIFuHD models

- **`user_study/`**: Resources and code related to analysisng the user study

- **`visuals/`**: Visualisation tools and scripts for generating graphs and graphics

### `images/`
Storage for source 4D-DRESS images.

### `label_files/`
Contains label data for the 4D-DRESS dataset scans.

### `obj_4ddress_files/`
4D-Dress model files in OBJ format.

### `obj_4ddress_labelled_files/`
Labelled 4D-Dress model files.

### `obj_pifuhd_aligned_files/`
Aligned PIFuHD model files.

### `obj_pifuhd_files/`
Original PIFuHD model files.

### `obj_pifuhd_labelled_files/`
Labelled PIFuHD model files.

### `pkl_meshes/`
Raw 4D-DRESS mesh data stored in pickle format.