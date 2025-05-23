- `BlurImageFast`: The BlurImageFast node provides a fast and efficient way to apply Gaussian blur to images. It is designed to blur images by specifying the radius of the blur in both the x and y directions, allowing for customizable blur effects.
    - Inputs:
        - `images` (Required): The 'images' parameter represents the images to be blurred. It is crucial for defining the input images on which the Gaussian blur effect will be applied. Type should be `IMAGE`.
        - `radius_x` (Required): The 'radius_x' parameter specifies the horizontal radius of the Gaussian blur. It determines the extent of blurring along the x-axis of the images. Type should be `INT`.
        - `radius_y` (Required): The 'radius_y' parameter specifies the vertical radius of the Gaussian blur. It determines the extent of blurring along the y-axis of the images. Type should be `INT`.
    - Outputs:
        - `image`: The output is a blurred version of the input images, achieved through Gaussian blurring as specified by the radius_x and radius_y parameters. Type should be `IMAGE`.
