- `LayerUtility_ ImageChannelMerge`: The ImageChannelMerge node is designed to merge separate image channels into a single image, supporting various color modes. It allows for the flexible combination of up to four image channels, accommodating optional fourth channel input for extended color space manipulation.
    - Inputs:
        - `channel_i` (Required): Represents one of the image channels to be merged. Each channel contributes to the composite image by adding its unique layer of detail, color, or depth. The index 'i' can range from 1 to 4, where channels 1 through 3 are required and channel 4 is optional, allowing for more complex color space manipulations when included. Type should be `IMAGE`.
        - `mode` (Required): Specifies the color mode for the merged image, such as RGBA, YCbCr, LAB, or HSV. This determines how the channels are combined and interpreted, affecting the overall appearance of the final composite image. Type should be `COMBO[STRING]`.
    - Outputs:
        - `image`: The resulting image after merging the specified channels. It represents a composite image in the specified color mode. Type should be `IMAGE`.
