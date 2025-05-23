- `LayerUtility_ HSV Value`: The 'LayerUtility: HSV Value' node is designed to convert color values into their corresponding HSV (Hue, Saturation, Value) components. It supports input in both hexadecimal string and RGB tuple formats, facilitating the transformation of color representations into a format that's widely used for color manipulation and analysis.
    - Inputs:
        - `color_value` (Required): The 'color_value' parameter accepts a color in either hexadecimal string format or RGB tuple format. It is crucial for determining the output HSV values, as the node converts this input into its HSV components. Type should be `*`.
    - Outputs:
        - `H`: Represents the Hue component of the color in the HSV color space, as an integer. Type should be `INT`.
        - `S`: Represents the Saturation component of the color in the HSV color space, as an integer. Type should be `INT`.
        - `V`: Represents the Value component of the color in the HSV color space, as an integer. Type should be `INT`.
