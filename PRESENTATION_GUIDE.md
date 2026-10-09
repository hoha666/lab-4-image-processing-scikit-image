# Lab 4 Presentation and Python Study Guide

This guide follows `notebooks/lab4_image_processing.ipynb`. Use the **What to tell the TA** parts as a short presentation script. Use the **Line-by-line explanation** parts to understand the Python.

## One-minute overview

> In this lab, we represent images as NumPy arrays and use scikit-image to perform a basic image-processing pipeline. We load and inspect RGB images, convert them to grayscale, resize and rescale an image, segment pixels using a threshold, and find a coin using template matching. Finally, we implement normalized cross-correlation ourselves and verify that it finds the same location as scikit-image.

Important vocabulary:

- **Pixel:** one picture element. A grayscale pixel has one intensity; an RGB pixel has red, green, and blue values.
- **Shape:** the dimensions of an array. RGB images normally have `(height, width, 3)`.
- **dtype:** the data type stored in an array, such as `uint8`, `float64`, or `bool`.
- **Intensity:** the brightness value of a pixel.
- **Grayscale:** an image with one brightness value per pixel instead of three color channels.
- **Segmentation:** dividing an image into meaningful regions, such as object and background.
- **Template matching:** searching for the location most similar to a smaller example image.

---

## Section 0 - Imports and setup

### What to tell the TA

> First, I import the libraries needed for arrays, plotting, image processing, paths, and two-dimensional correlation. I make the notebook independent of the current working directory, create an `images` directory if necessary, and provide the standard coins and astronaut images as JPEG files. This makes the project reproducible on another computer.

### Line-by-line explanation

1. `from pathlib import Path`
   - Imports the `Path` class, which provides a clean, operating-system-independent way to work with folders and filenames.

2. `import os`
   - Imports Python's operating-system utilities. The notebook retains this import to reflect the lab setup, although `Path` handles the paths in our implementation.

3. `import numpy as np`
   - Imports NumPy and gives it the conventional short name `np`. Images are NumPy arrays, and later we use NumPy for calculations and array indexing.

4. `import matplotlib.pyplot as plt`
   - Imports Matplotlib's plotting interface as `plt`. We use it to display images, histograms, markers, and rectangles.

5. `import skimage`
   - Imports the top-level scikit-image package. We use it here to print the installed library version.

6. `from skimage import data, io`
   - Imports `data`, which contains sample images, and `io`, which reads and writes image files.

7. `from skimage.color import rgb2gray, gray2rgb`
   - Imports functions for converting RGB to grayscale and grayscale back to three-channel RGB.

8. `from skimage.transform import rescale, resize`
   - Imports the two size-changing functions used in Task 3.

9. `from skimage.feature import match_template`
   - Imports scikit-image's ready-made normalized-correlation template matcher.

10. `from scipy.signal import correlate2d`
    - Imports efficient two-dimensional correlation, which is used as a building block in our own template matcher.

11. `plt.rcParams["figure.figsize"] = (10, 5)`
    - Changes Matplotlib's default figure size to 10 inches wide and 5 inches high.

12. `plt.rcParams["image.cmap"] = "gray"`
    - Makes grayscale the default color map when a two-dimensional image is displayed.

13. `cwd = Path.cwd()`
    - Gets the folder from which the notebook is currently running. `cwd` means current working directory.

14. `PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd`
    - This is a conditional expression. If the notebook runs inside the `notebooks` folder, the project root is its parent; otherwise, the current folder is treated as the root.

15. `IMAGE_DIR = PROJECT_ROOT / "images"`
    - Uses Path's `/` operator to join the project root with the `images` folder name.

16. `IMAGE_DIR.mkdir(exist_ok=True)`
    - Creates the image directory. `exist_ok=True` prevents an error if it already exists.

17. `coins_path = IMAGE_DIR / "coins.jpg"`
    - Constructs the full path to `coins.jpg`.

18. `astronaut_path = IMAGE_DIR / "astronaut.jpg"`
    - Constructs the full path to `astronaut.jpg`.

19. `if not coins_path.exists():`
    - Checks whether the coins file is missing. The indented line executes only if the condition is true.

20. `io.imsave(coins_path, gray2rgb(data.coins()))`
    - Loads scikit-image's grayscale coins sample, converts it to three RGB channels, and saves it as a JPEG. The RGB conversion makes pixel access such as `[1, 100, 1]` valid.

21. `if not astronaut_path.exists():`
    - Checks whether the astronaut file is missing.

22. `io.imsave(astronaut_path, data.astronaut())`
    - Loads and saves scikit-image's RGB astronaut sample.

23. `print(f"scikit-image version: {skimage.__version__}")`
    - Prints the library version. The leading `f` creates an f-string, so the expression inside braces is evaluated.

24. `print(f"Images: {IMAGE_DIR.resolve()}")`
    - Prints the absolute image-directory path. `resolve()` converts the path to its complete form.

---

## Section 1 - Load, inspect, and visualize images

### What to tell the TA

> An image becomes a NumPy array when it is loaded. I inspect its shape to understand its dimensions, its dtype to understand how intensities are stored, and one pixel to demonstrate array indexing. Both images are RGB, so each pixel has three channel values. I then display both images side by side.

If asked about `[1, 100, 1]`, say:

> NumPy uses zero-based indexing. This selects row 1, column 100, and channel 1. RGB channels are numbered 0, 1, and 2, so channel 1 is green.

### Line-by-line explanation

1. `coins = io.imread(coins_path)`
   - Reads the coins JPEG and stores its pixel array in `coins`.

2. `astronaut = io.imread(astronaut_path)`
   - Reads the astronaut JPEG into another NumPy array.

3. `fig, axes = plt.subplots(1, 2, figsize=(11, 5))`
   - Creates one figure containing one row and two plotting areas. `fig` is the full figure; `axes` holds the two individual axes.

4. `for ax, image, name in zip(axes, [coins, astronaut], ["coins.jpg", "astronaut.jpg"]):`
   - `zip` combines the corresponding axis, image, and filename. The `for` loop processes both images without duplicating code.

5. `print(f"{name}: shape={image.shape}, dtype={image.dtype}, pixel[1,100,1]={image[1,100,1]}")`
   - Prints the image name, dimensions, storage type, and selected green-channel intensity.

6. `ax.imshow(image)`
   - Displays the current image on the current axis.

7. `ax.set_title(name)`
   - Places the filename above the image.

8. `ax.axis("off")`
   - Hides coordinate ticks and borders because they are unnecessary for this image view.

9. `plt.tight_layout()`
   - Automatically adjusts spacing so the plots and titles do not overlap.

10. `plt.show()`
    - Renders the completed figure in the notebook.

---

## Section 2 - Convert RGB to grayscale

### What to tell the TA

> I convert each RGB image to grayscale with `rgb2gray`. The spatial dimensions remain the same, but the three-channel dimension disappears because each pixel now has one luminance value. The JPEG input uses `uint8` values from 0 to 255. The grayscale result uses floating-point values from 0 to 1. Grayscale simplifies later operations because we compare one intensity per pixel instead of three colors.

Important detail: grayscale is not simply an average of R, G, and B. The conversion uses a weighted luminance calculation because human vision does not perceive all colors equally.

### Line-by-line explanation

1. `coins_gray = rgb2gray(coins)`
   - Converts the coins RGB array into a two-dimensional grayscale array.

2. `astronaut_gray = rgb2gray(astronaut)`
   - Performs the same conversion for the astronaut.

3. `for name, original, grayscale in [...]`
   - Loops over two tuples. Each tuple contains a label, the original image, and its grayscale version.

4. `print(f"{name:9s}: ...")`
   - Prints a formatted comparison. `:9s` reserves nine characters for the name so the output aligns neatly.

5. `{original.shape}, {original.dtype} -> {grayscale.shape}, {grayscale.dtype}`
   - Shows the change from RGB shape and integer dtype to grayscale shape and floating-point dtype.

6. `grayscale.min()` and `grayscale.max()`
   - Return the darkest and brightest values. `:.3f` prints each with three digits after the decimal point.

7. `fig, axes = plt.subplots(1, 2, figsize=(11, 5))`
   - Creates two side-by-side axes for the grayscale results.

8. `axes[0].imshow(coins_gray, cmap="gray")`
   - Displays coins on the first axis. `cmap="gray"` maps low values to black and high values to white.

9. `axes[0].set_title("Coins - grayscale")`
   - Adds the first title.

10. `axes[1].imshow(astronaut_gray, cmap="gray")`
    - Displays the grayscale astronaut on the second axis.

11. `axes[1].set_title("Astronaut - grayscale")`
    - Adds the second title.

12. `for ax in axes: ax.axis("off")`
    - A compact one-line loop that hides axes on both plots.

13. `plt.tight_layout()` and `plt.show()`
    - Correct the spacing and render the figure.

---

## Section 3 - Resize and rescale

### What to tell the TA

> `resize` and `rescale` both change image dimensions, but they express the target differently. `resize` accepts exact output dimensions, so I choose 300 by 200 pixels. `rescale` accepts a multiplier, so factors 0.75, 0.5, and 0.25 preserve the original aspect ratio. I enable anti-aliasing to reduce jagged edges and sampling artifacts during downscaling.

If asked why the shapes still contain `3`, say:

> `channel_axis=-1` tells `rescale` that the final axis contains color channels. It rescales height and width without treating RGB as a spatial dimension.

### Line-by-line explanation

1. `astronaut_resized = resize(astronaut, (300, 200), anti_aliasing=True)`
   - Produces an image with exactly 300 rows and 200 columns. The RGB channel is preserved automatically.

2. `scale_factors = [0.75, 0.5, 0.25]`
   - Creates a list of the three required scale multipliers.

3. `astronaut_rescaled = { ... }`
   - Creates a dictionary. Each scale factor becomes a key and its rescaled image becomes the value.

4. `factor: rescale(astronaut, factor, channel_axis=-1, anti_aliasing=True)`
   - Rescales the image by the current factor, marks the last axis as RGB channels, and enables anti-aliasing.

5. `for factor in scale_factors`
   - This dictionary comprehension repeats the rescaling operation for all three factors.

6. `print(f"Original: {astronaut.shape}")`
   - Prints the input dimensions for comparison.

7. `print(f"Resized to chosen dimensions: {astronaut_resized.shape}")`
   - Confirms the exact resize result.

8. `for factor, image in astronaut_rescaled.items():`
   - Loops over every key-value pair in the dictionary.

9. `print(f"Rescaled by {factor}: {image.shape}")`
   - Shows how each multiplier changes the shape.

10. `fig, axes = plt.subplots(1, 4, figsize=(15, 5))`
    - Creates four axes: one for the exact resize and three for the scale factors.

11. `axes[0].imshow(astronaut_resized)`
    - Displays the exact-size result.

12. `axes[0].set_title("Resize: 300 x 200")`
    - Labels the exact-size result.

13. `for ax, (factor, image) in zip(axes[1:], astronaut_rescaled.items()):`
    - Pairs the remaining three axes with the three dictionary entries. `axes[1:]` means all axes starting at index 1.

14. `ax.imshow(image)` and `ax.set_title(f"Rescale: {factor}")`
    - Display and label each scaled image.

15. The final loop, `tight_layout`, and `show`
    - Hide the axes, improve spacing, and display the figure.

---

## Section 4 - Histogram and threshold segmentation

### What to tell the TA

> A histogram counts how many pixels occur at each intensity. Because the grayscale image is `float64`, valid intensities are between 0 and 1. I inspect the distribution and select `t = 0.42` as a useful manual boundary. The comparison `coins_gray < t` creates a Boolean mask: pixels darker than 0.42 become `True`, and the rest become `False`. This is basic segmentation because it separates pixels into two classes.

If asked whether 0.42 is the only correct threshold, say:

> No. It is a manually selected value based on the histogram. A different value changes the segmentation. An automatic method such as Otsu's method could select a threshold algorithmically, but this task specifically asks us to inspect the histogram and choose `t`.

### Line-by-line explanation

1. `t = 0.42`
   - Stores the chosen threshold in a variable named `t`.

2. `binary = coins_gray < t`
   - Performs an element-by-element comparison. The result has the same height and width but stores only `True` or `False`.

3. `fig, axes = plt.subplots(1, 3, figsize=(15, 4))`
   - Creates axes for the grayscale image, histogram, and binary result.

4. `axes[0].imshow(coins_gray, cmap="gray")`
   - Displays the grayscale input.

5. `axes[0].set_title(...)` and `axes[0].axis("off")`
   - Add a title and hide coordinate decoration.

6. `coins_gray.ravel()`
   - Flattens the two-dimensional image into a one-dimensional sequence so all pixel intensities can be passed to the histogram.

7. `axes[1].hist(coins_gray.ravel(), bins=256)`
   - Groups all intensity values into 256 bins and plots their frequencies.

8. `axes[1].axvline(t, color="red", linestyle="--", label=f"t = {t}")`
   - Draws a dashed red vertical line showing the chosen threshold on the histogram.

9. `axes[1].set(title=..., xlabel=..., ylabel=...)`
   - Sets the histogram title and axis labels in one call.

10. `axes[1].legend()`
    - Displays the label for the red threshold line.

11. `axes[2].imshow(binary, cmap="gray")`
    - Displays the Boolean mask. Matplotlib renders `False` as black and `True` as white.

12. `axes[2].set_title(f"coins_gray < {t}")`
    - Labels the result with the exact comparison used.

13. `plt.tight_layout()` and `plt.show()`
    - Arrange and display the three panels.

14. `print(f"grayscale dtype={coins_gray.dtype}; binary dtype={binary.dtype}")`
    - Confirms that thresholding converts a floating-point intensity image into a Boolean mask.

---

## Section 5 - Template matching with scikit-image

### What to tell the TA

> Template matching searches the large image for the position that most closely resembles a smaller template. I crop one coin from rows 170 to 219 and columns 75 to 129. `match_template` computes a normalized correlation score for every valid position. I find the largest score, convert its array index into x-y coordinates, and draw a red rectangle around the best match. The result is `(x=75, y=170)` with a score of 1 because the exact crop is present in the source image.

Remember the coordinate distinction:

- NumPy indexes arrays as `[row, column]`, which means `[y, x]`.
- Matplotlib positions shapes as `(x, y)`.
- That is why the code reverses the index order.

### Line-by-line explanation

1. `template_image = data.coins()`
   - Loads the original two-dimensional grayscale coins sample.

2. `coin = template_image[170:220, 75:130]`
   - Crops rows 170 through 219 and columns 75 through 129. Python stops before the ending index, so the crop is 50 by 55 pixels.

3. `result = match_template(template_image, coin)`
   - Slides the template over the image and returns a two-dimensional response map of similarity scores.

4. `np.argmax(result)`
   - Returns the flattened index of the largest value in the response map.

5. `np.unravel_index(np.argmax(result), result.shape)`
   - Converts that flattened index back into the response map's `(row, column)` coordinates.

6. `ij = ...`
   - Stores the `(row, column)` pair.

7. `x, y = ij[::-1]`
   - `[::-1]` reverses the pair from `(row, column)` to `(column, row)`, which corresponds to `(x, y)`.

8. `fig = plt.figure(figsize=(12, 4))`
   - Creates the overall figure.

9. `ax1`, `ax2`, and `ax3 = plt.subplot(1, 3, ...)`
   - Create three axes in a one-row, three-column layout.

10. `ax1.imshow(coin)` and `ax1.set_title("Template")`
    - Display and label the small coin template.

11. `ax2.imshow(template_image)` and `ax2.set_title("Image")`
    - Display and label the complete search image.

12. `hcoin, wcoin = coin.shape`
    - Unpacks the template's height and width into separate variables.

13. `plt.Rectangle((x, y), wcoin, hcoin, ...)`
    - Creates an unfilled red rectangle at the best top-left position with the same dimensions as the template.

14. `ax2.add_patch(...)`
    - Adds that rectangle to the full image.

15. `ax3.imshow(result)`
    - Displays the correlation response map. Bright values indicate stronger similarity.

16. `ax3.plot(x, y, "o", ...)`
    - Draws an unfilled red circle at the maximum response.

17. The title, axis loop, `tight_layout`, and `show`
    - Label the result, remove plot decoration, arrange the figure, and render it.

18. `result[y, x]`
    - Retrieves the best score. Array access is `[y, x]`, even though the plotted point is `(x, y)`.

19. `:.6f`
    - Formats the score with six digits after the decimal point.

---

## Section 6 - Our normalized cross-correlation implementation

### What to tell the TA

> In the final task, I implement normalized cross-correlation, or NCC, without calling `match_template`. At each possible template location, NCC removes the mean brightness from the template and image patch, computes their correlation, and divides by their energies. Normalization makes the comparison less sensitive to uniform brightness and contrast changes. I use `correlate2d` to perform the sliding sums efficiently. My implementation finds `(x=75, y=170)` with score 1, exactly matching scikit-image.

The NCC idea is:

`score = similarity numerator / normalization denominator`

- A score close to `1` means a strong positive match.
- A score close to `0` means little linear similarity.
- A score close to `-1` means opposite intensity patterns.

### Function: validation and preparation

1. `def normalized_cross_correlation(image, template):`
   - Defines a reusable function with two input parameters.

2. `"""Return valid normalized cross-correlation ..."""`
   - A docstring describing the function. `valid` means only positions where the whole template fits inside the image are returned.

3. `image = np.asarray(image, dtype=np.float64)`
   - Converts the input into a NumPy array of 64-bit floating-point values. Floating point is needed for means, subtraction, division, and square roots.

4. `template = np.asarray(template, dtype=np.float64)`
   - Performs the same safe conversion for the template.

5. `if image.ndim != 2 or template.ndim != 2:`
   - Checks that both inputs are two-dimensional grayscale arrays. `ndim` is the number of dimensions and `or` means either invalid condition is enough.

6. `raise ValueError("image and template must both be 2-D")`
   - Stops the function with a clear error message if the inputs are invalid.

7. `zip(template.shape, image.shape)`
   - Pairs template height with image height and template width with image width.

8. `any(t > i for t, i in ...)`
   - Tests whether any template dimension is larger than its matching image dimension.

9. `raise ValueError("template must not be larger than image")`
   - Prevents an impossible search.

### Function: template statistics

10. `h, w = template.shape`
    - Stores template height and width.

11. `n = h * w`
    - Calculates the total number of template pixels.

12. `centered_template = template - template.mean()`
    - Subtracts the template's average intensity from every template pixel, producing a zero-mean template.

13. `template_energy = np.sum(centered_template ** 2)`
    - Squares every centered value and adds them. This measures the template's intensity variation or energy.

### Function: sliding image statistics and NCC

14. `numerator = correlate2d(image, centered_template, mode="valid")`
    - Slides the centered template across the image and calculates the sum of products at every fully valid position. Because the template has zero mean, this is the NCC numerator.

15. `kernel = np.ones((h, w), dtype=np.float64)`
    - Creates an all-ones array with the template's dimensions. It is used to calculate sums for every image patch.

16. `patch_sum = correlate2d(image, kernel, mode="valid")`
    - Calculates the pixel sum of every image patch that has the same size as the template.

17. `patch_sum_sq = correlate2d(image ** 2, kernel, mode="valid")`
    - Squares the image values first, then calculates the sum of squares for every patch.

18. `patch_sum_sq - (patch_sum ** 2) / n`
    - Uses the identity `sum(x^2) - sum(x)^2 / n` to calculate each patch's zero-mean energy without explicitly constructing every patch.

19. `np.maximum(..., 0)`
    - Clips tiny negative values caused by floating-point rounding to zero. True energy cannot be negative.

20. `denominator = np.sqrt(patch_energy * template_energy)`
    - Computes the normalization factor from the image-patch energy and template energy.

21. `np.divide(numerator, denominator, ...)`
    - Divides the numerator by the denominator to obtain NCC scores.

22. `out=np.zeros_like(numerator)`
    - Creates a zero-filled output array with the same shape as the numerator.

23. `where=denominator > 1e-12`
    - Divides only where the denominator is safely nonzero. This avoids division-by-zero warnings in constant patches.

24. `return ...`
    - Sends the completed response map back to the caller.

### Calling and visualizing our function

25. `own_result = normalized_cross_correlation(template_image, coin)`
    - Runs our matcher using the same full image and template used in Task 5.

26. `own_y, own_x = np.unravel_index(np.argmax(own_result), own_result.shape)`
    - Finds the largest score and directly stores its row as `own_y` and column as `own_x`.

27. `fig, axes = plt.subplots(1, 3, figsize=(12, 4))`
    - Creates panels for the template, matched image, and NCC response map.

28. `axes[0].imshow(coin)` and its title
    - Display the search template.

29. `axes[1].imshow(template_image)`
    - Displays the full image.

30. `axes[1].add_patch(plt.Rectangle(...))`
    - Marks our best match with a red rectangle.

31. `axes[2].imshow(own_result)`
    - Displays our NCC response map.

32. `axes[2].plot(own_x, own_y, "o", ...)`
    - Marks the maximum response with a red circle.

33. `for ax in axes: ax.axis("off")`
    - Removes plot axes from all three panels.

34. `plt.tight_layout()` and `plt.show()`
    - Arrange and render the figure.

35. The first `print(...)`
    - Reports our best coordinates and score.

36. The second `print(...)`
    - Reports scikit-image's coordinates for comparison.

37. `print(f"Same best location: {(own_x, own_y) == (x, y)}")`
    - Compares the two coordinate tuples. It prints `True`, demonstrating that both implementations agree.

---

## Likely TA questions and short answers

### Why use NumPy arrays for images?

Images form regular grids of numeric values. NumPy provides efficient storage, indexing, element-wise comparisons, and mathematical operations on those grids.

### What is the difference between `uint8`, `float64`, and `bool`?

- `uint8` stores whole numbers from 0 to 255, common in image files.
- `float64` stores decimal values; scikit-image commonly represents processed intensities from 0 to 1.
- `bool` stores only `True` or `False`, suitable for a binary segmentation mask.

### Why convert to grayscale?

It reduces every pixel from three values to one intensity, which simplifies histograms, thresholding, and many matching algorithms.

### Resize versus rescale?

`resize` specifies exact output dimensions. `rescale` specifies a multiplication factor and normally preserves the aspect ratio.

### Why use anti-aliasing?

When reducing image size, many source pixels must become fewer output pixels. Anti-aliasing filters high-frequency detail first, reducing jagged edges and moire patterns.

### What does the histogram show?

The x-axis represents intensity and the y-axis represents the number of pixels with that intensity. Peaks can represent common regions such as background and objects.

### Why is thresholding called segmentation?

It assigns pixels to separate classes. Here the two classes are below-threshold and above-threshold pixels.

### Why is the best template score exactly 1?

The template was cropped directly from the same image. At its original location, the template and patch are identical, so normalized correlation reaches its maximum value.

### Why normalize correlation?

Raw correlation is affected by overall brightness and contrast. Subtracting means and dividing by energy makes the score measure pattern similarity more fairly.

### Did we really implement our own matcher?

Yes. Task 5 uses scikit-image's `match_template`. Task 6 implements the NCC formula, input validation, patch statistics, safe normalization, maximum search, and visualization. SciPy's `correlate2d` is used only as an efficient primitive for repeated sliding sums and products.

---

## Suggested presentation order

1. State that an image is a NumPy array and show shape, dtype, and pixel indexing.
2. Show RGB-to-grayscale conversion and explain the shape and dtype changes.
3. Compare exact resizing with factor-based rescaling and mention anti-aliasing.
4. Point to the histogram, the red threshold line, and the Boolean segmentation.
5. Show the coin template, response map, and best-match rectangle.
6. Explain the NCC numerator and denominator at a high level.
7. Finish by showing that both implementations return `(x=75, y=170)`.

## Final conclusion to memorize

> This lab taught me that image processing is mainly numerical array processing. I learned how image shape and dtype affect operations, how grayscale simplifies analysis, how resizing differs from rescaling, how a histogram supports threshold selection, and how normalized cross-correlation locates a known pattern. Implementing NCC myself also helped me understand what `match_template` does internally rather than treating it as a black box.
