# Lab 4: Getting Started with Image Processing Using scikit-image

This repository implements all tasks in the Lab 4 assignment in a Jupyter notebook.

## Contents

- `notebooks/lab4_image_processing.ipynb` - completed, executable lab report
- `images/coins.jpg` and `images/astronaut.jpg` - created automatically from scikit-image sample data
- `build_notebook.py` - reproducibly generates the notebook source

## Run

```powershell
python -m pip install -r requirements.txt
jupyter notebook notebooks/lab4_image_processing.ipynb
```

Run all cells from top to bottom. The first code cell creates the image files if they are missing.

## Tasks covered

0. Software setup
1. Load, inspect, and visualize images
2. Convert RGB images to grayscale and explain shape/dtype changes
3. Resize and rescale the astronaut image
4. Threshold the grayscale coins image using its histogram
5. Match a coin template with `skimage.feature.match_template`
6. Implement normalized cross-correlation template matching independently
