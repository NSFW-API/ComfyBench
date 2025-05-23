- `ImageRemoveAlpha+`: This node is designed to process images by removing the alpha channel, effectively converting images with transparency into a standard RGB format. It ensures that images are compatible with operations or environments that do not support or require transparency.
    - Inputs:
        - `image` (Required): The input image with a potential alpha channel. This parameter is crucial for the operation as it determines whether the alpha channel, if present, needs to be removed to convert the image into RGB format. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output image after the alpha channel has been removed, provided in standard RGB format. Type should be `IMAGE`.
