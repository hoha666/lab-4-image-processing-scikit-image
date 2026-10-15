# Lab 4 Presentation Guide

This guide matches the current `notebooks/lab4_image_processing.ipynb`. The code and explanations follow Tasks 0-6 in the PDF.

## Short introduction

> In this lab, images are handled as NumPy arrays. I load and inspect the supplied coins and astronaut images, prepare grayscale versions, resize and rescale the astronaut, segment the coins with a threshold, apply scikit-image template matching, and implement my own simple template matcher.

## Task 0 - Software setup

### What to tell the TA

> The lab requires scikit-image. The required packages are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

### Setup code

1. `import os`
   - Provides `os.path.join`, which joins a folder path and filename.

2. `import numpy as np`
   - Imports NumPy for arrays and calculations.

3. `import matplotlib.pyplot as plt`
   - Imports plotting functions for images and histograms.

4. `import skimage`
   - Imports the scikit-image package required by the lab.

5. `from skimage import data, io`
   - `data` provides the Task 5 coins example and `io` reads the supplied JPEG files.

6. `from skimage.color import rgb2gray`
   - Imports the RGB-to-grayscale conversion function.

7. `from skimage.transform import rescale, resize`
   - Imports the two size-changing functions required in Task 3.

8. `from skimage.feature import match_template`
   - Imports scikit-image's template matcher for Task 5.

9. `from skimage.util import img_as_float`
   - Converts the already-grayscale coins image from `uint8` to floating-point form.

10. `filename_path = "../images"`
    - Points from the `notebooks` folder to the repository's `images` folder.

11. `os.path.join(...)`
    - Creates the complete paths to `coins.jpg` and `astronaut.jpg`.

---

## Task 1 - Load and visualize images

### What to tell the TA

> I load both supplied images with `io.imread`. An image becomes a NumPy array, so I can inspect its shape, dtype, and pixel values. The coins image is already grayscale and has shape `(303, 384)`. The astronaut is RGB and has shape `(512, 512, 3)`.

The last `3` in `(512, 512, 3)` means that every astronaut pixel contains red, green, and blue values.

### Line-by-line explanation

1. `coins = io.imread(coins_path)`
   - Reads `coins.jpg` into a NumPy array.

2. `print("Coins shape:", coins.shape)`
   - Prints `(303, 384)`: 303 rows and 384 columns.

3. `print("Coins dtype:", coins.dtype)`
   - Prints `uint8`, meaning integer intensities from 0 to 255.

4. `coins[1, 100]`
   - Gets the grayscale intensity at row 1 and column 100.

5. `plt.imshow(coins, cmap="gray")`
   - Displays the coins image in grayscale.

6. `plt.axis("off")` and `plt.show()`
   - Hide the plot axes and display the image.

7. `astronaut = io.imread(astronaut_path)`
   - Reads `astronaut.jpg` into a NumPy array.

8. `print("Astronaut shape:", astronaut.shape)`
   - Prints `(512, 512, 3)`: height, width, and three RGB channels.

9. `print("Astronaut dtype:", astronaut.dtype)`
   - Prints `uint8`.

10. `astronaut[1, 100, 1]`
    - Gets the green-channel value at row 1 and column 100. Channel 0 is red, 1 is green, and 2 is blue.

11. `plt.imshow(astronaut)`
    - Displays the RGB image in color.

---

## Task 2 - Color-space conversion

### What to tell the TA

> The coins file is already grayscale, so `img_as_float` changes only its dtype and intensity range. The astronaut is RGB, so `rgb2gray` removes its color-channel dimension and calculates one grayscale intensity per pixel.

### Answer required by the PDF

- Coins before: shape `(303, 384)`, dtype `uint8`, values 0-255.
- Coins after: shape `(303, 384)`, dtype `float64`, values 0-1.
- Its shape is unchanged because it was already grayscale.
- Astronaut before: shape `(512, 512, 3)`, dtype `uint8`, values 0-255.
- Astronaut after: shape `(512, 512)`, dtype `float64`, values 0-1.
- The final dimension disappears because three RGB values become one grayscale intensity.

### Line-by-line explanation

1. `coins_gray = img_as_float(coins)`
   - Converts coins from `uint8` values to floating-point values without changing its shape.

2. `astronaut_gray = rgb2gray(astronaut)`
   - Converts the three astronaut color channels into one grayscale channel.

3. The four `print(...)` lines
   - Show the shape and dtype before and after each conversion.

4. `plt.imshow(coins_gray, cmap="gray")`
   - Displays the grayscale coins result.

5. `plt.imshow(astronaut_gray, cmap="gray")`
   - Displays the grayscale astronaut result.

---

## Task 3 - Image rescale and resize

### What to tell the TA

> Following the PDF example, I use the grayscale astronaut from Task 2. I resize it to a chosen shape of 128 by 128. I also rescale it by the required factors 0.75, 0.5, and 0.25.

### Observation required by the PDF

`resize` receives the exact output shape. `rescale` receives a factor that is multiplied by the original dimensions.

- Original: `(512, 512)`
- Resize to chosen dimensions: `(128, 128)`
- Rescale by 0.75: `(384, 384)`
- Rescale by 0.50: `(256, 256)`
- Rescale by 0.25: `(128, 128)`

The resize and 0.25 rescale have the same shape in this example, but they were specified differently.

### Line-by-line explanation

1. `new_height = astronaut_gray.shape[0] // 4`
   - Divides the original height by 4: 512 becomes 128.

2. `new_width = astronaut_gray.shape[1] // 4`
   - Divides the original width by 4.

3. `resize(astronaut_gray, (new_height, new_width), anti_aliasing=True)`
   - Produces the exact chosen shape. Anti-aliasing reduces visual artifacts while shrinking.

4. `rescale(astronaut_gray, 0.75, anti_aliasing=False)`
   - Multiplies both dimensions by 0.75.

5. The next two `rescale` lines
   - Repeat the operation with factors 0.50 and 0.25.

6. The five `print(...)` lines
   - Show the original and resulting shapes.

7. Each `plt.imshow(..., cmap="gray")`
   - Displays one result so the size and detail can be observed.

---

## Task 4 - Image thresholding

### What to tell the TA

> I use the grayscale coins image from Task 2. First, I check its dtype and plot a 256-bin histogram. Because the image is `float64`, its values are in the range 0-1. I select `t = 0.42` from the histogram and create the binary image with `coins_gray < t`.

### Answer required by the PDF

The threshold must match the image dtype. A threshold such as 100 would suit some `uint8` images, but it would be incorrect here because `coins_gray` is `float64` in the range 0-1. Therefore, `t = 0.42` is used. Pixels below 0.42 become `True`; pixels at or above 0.42 become `False`.

### Line-by-line explanation

1. `print("Grayscale dtype:", coins_gray.dtype)`
   - Confirms that the image is `float64`.

2. `coins_gray.ravel()`
   - Flattens the 2-D image into one sequence of pixel values.

3. `plt.hist(..., bins=256)`
   - Calculates and displays a histogram with 256 groups.

4. `t = 0.42`
   - Stores the selected threshold.

5. `binary = coins_gray < t`
   - Compares every pixel with the threshold and returns `True` or `False`.

6. `plt.imshow(binary, cmap="gray")`
   - Displays the Boolean result as a black-and-white image.

---

## Task 5 - Template matching

### What to tell the TA

> I follow the PDF's sample code. I crop one coin from scikit-image's coins image and use it as the template. `match_template` compares it with every valid location in the full image. The largest score gives the best match.

### Explanation required by the PDF

The red rectangle shows the best-matching region in the full image. The red circle marks the same location in the result image. The best position is `(x=75, y=170)`, which is expected because the template was cropped from that exact location.

### Line-by-line explanation

1. `image = data.coins()`
   - Loads scikit-image's grayscale coins data used in the PDF example.

2. `coin = image[170:220, 75:130]`
   - Crops a 50-by-55-pixel coin template.

3. `result = match_template(image, coin)`
   - Creates a result array containing a match score for every valid position.

4. `np.argmax(result)`
   - Finds the position of the largest match score.

5. `np.unravel_index(..., result.shape)`
   - Changes the single maximum index into row and column coordinates.

6. `x, y = ij[::-1]`
   - Reverses `(row, column)` into plotting coordinates `(x, y)`.

7. `plt.subplot(1, 3, ...)`
   - Creates panels for the template, full image, and result.

8. `plt.Rectangle((x, y), wcoin, hcoin, ...)`
   - Creates the red rectangle at the best position.

9. `ax2.add_patch(rectangle)`
   - Adds the rectangle to the full image.

10. `ax3.plot(x, y, "o", ...)`
    - Adds the red circle to the result image.

---

## Task 6 - Own template-matching implementation

### What to tell the TA

> I implement my own simple template matcher using sum of squared differences, or SSD. I use the same full image and coin template as Task 5. The template moves over the image one position at a time. The smallest SSD score is the best match.

### How SSD works

For each position:

1. Crop an image patch with the same shape as the template.
2. Subtract the template from that patch.
3. Square every difference.
4. Add the squared differences.

Identical images have a score of zero, so `np.argmin` finds the best position.

### Line-by-line explanation

1. `def own_template_matching(image, template):`
   - Defines our own reusable matching function.

2. `image.astype(float)` and `template.astype(float)`
   - Convert integer pixels to floating point so subtraction is safe. Without this, unsigned integers could wrap around.

3. `template_height, template_width = template.shape`
   - Stores the template dimensions.

4. `result_height` and `result_width`
   - Calculate how many complete template positions fit inside the image.

5. `scores = np.zeros(...)`
   - Creates an array in which every SSD score will be stored.

6. `for y in range(result_height):`
   - Moves the template down through every valid row.

7. `for x in range(result_width):`
   - Moves the template across every valid column.

8. `patch = image[y:y + template_height, x:x + template_width]`
   - Crops the current image patch.

9. `difference = patch - template`
   - Calculates the pixel-by-pixel difference.

10. `scores[y, x] = np.sum(difference ** 2)`
    - Squares and adds all differences, then stores the SSD score.

11. `return scores`
    - Returns the completed score image.

12. `np.argmin(own_result)`
    - Finds the smallest score because smaller SSD means a better match.

13. `np.unravel_index(...)`
    - Converts the position into `y` and `x` coordinates.

14. The rectangle and circle
    - Show the best position in the original image and score image.

15. The final comparison
    - Confirms that our matcher finds the same `(x=75, y=170)` position as Task 5.

---

## Suggested presentation order

1. Explain that images are NumPy arrays.
2. Show the Task 1 shapes, dtypes, and pixel indexing.
3. Explain the Task 2 before-and-after shape and dtype changes.
4. Compare `resize` with `rescale` in Task 3.
5. Explain why `t = 0.42` matches the Task 4 `float64` dtype.
6. Explain the Task 5 rectangle and result maximum.
7. Explain Task 6 SSD and why the minimum score is selected.

## Conclusion to memorize

> This lab taught me how images are represented as NumPy arrays and how shape and dtype affect image processing. I converted images to grayscale, changed image dimensions using resize and rescale, segmented an image using a histogram-based threshold, used scikit-image template matching, and implemented my own simple SSD template matcher.
