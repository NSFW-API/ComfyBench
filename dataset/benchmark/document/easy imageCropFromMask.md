- `easy imageCropFromMask`: The `imageCropFromMask` node is designed for cropping images based on a provided mask. It utilizes the mask to determine the relevant area of the image to retain, effectively isolating and extracting the portion of the image that corresponds to the mask's coverage.
    - Inputs:
        - `image` (Required): The `image` parameter represents the image to be cropped. It is essential for determining the area of interest that will be retained after the cropping process, based on the mask's outline. Type should be `IMAGE`.
        - `mask` (Required): The `mask` parameter is crucial for defining the area of the image to be cropped. It acts as a guide for isolating the specific portion of the image that should be retained, based on the mask's coverage. Type should be `MASK`.
        - `image_crop_multi` (Required): This parameter specifies the multiplier to adjust the crop size relative to the mask's bounding box, affecting the final cropped image size. Type should be `FLOAT`.
        - `mask_crop_multi` (Required): This parameter adjusts the size of the mask used for cropping, influencing the area of the image that gets cropped. Type should be `FLOAT`.
        - `bbox_smooth_alpha` (Required): This parameter controls the smoothing of bounding box size changes over time, affecting the consistency of the crop dimensions. Type should be `FLOAT`.
    - Outputs:
        - `crop_image`: The cropped portion of the original image, as determined by the mask and cropping parameters. Type should be `IMAGE`.
        - `crop_mask`: The cropped portion of the mask, corresponding to the cropped area of the image. Type should be `MASK`.
        - `bbox`: The bounding box coordinates of the cropped area within the original image. Type should be `BBOX`.
