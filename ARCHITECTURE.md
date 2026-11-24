# Architecture analysis

Run the analysis on a binary PNG with the plant in white and the background in
black:

```text
python analyze_architecture.py test_overlay.png --no-preview --drawing architecture.png
python analyze_architecture.py test_overlay.png --image original_photo.tif --drawing architecture.png
```

The program prints JSON. Without `--no-preview`, it also displays the existing
leaflet preview. `--drawing` saves a separate black-background drawing: white
is the plant skeleton, red is the estimated main axis, and green dots and
labels identify estimated branch junctions. Running the script with no
arguments retains the existing test-image workflow, including saving
`test_architecture.png` and `test_architecture_results.json`.

The main axis is the skeleton path from the lowest to the highest skeleton
pixel, so it represents an estimated **stem**, not an underground root. Its
length and branch junction positions are path lengths in original-image
pixels, approximated from a skeleton reduced to at most 1000 pixels on its
longest side. Consecutive branch spacings are differences between junction
positions along that path, ordered from base to tip; they include branches on
either side and can be zero. Branch IDs match the leaflet preview. Existing
leaflet circularities and the whole-plant external-outline circularity use
`4*pi*area/perimeter^2`; holes are excluded from the whole-plant outline.

An optional `--image` supplies the original, unaltered color image, aligned
pixel-for-pixel with the mask and of the same dimensions. The reported
`main_axis_color.mean_rgb` is the mean RGB of foreground pixels sampled along
the estimated axis centerline (0-255); it is null without a color image.
The axis may pass through leaves or junctions, so this is not a pure stem
color measurement. A binary mask alone cannot provide color.

These are experimental 2D image estimates, not calibrated physical lengths or
verified biological traits. Segmentation, overlap, orientation, skeleton
downsampling, and incorrect junction or axis detection can affect the results;
inspect the drawing and preview before interpreting them.
