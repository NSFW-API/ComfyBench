- `CR Color Bars`: The CR Color Bars node is designed to generate customizable color bar patterns. It allows users to specify various parameters such as color, orientation, and frequency to create visually appealing bar patterns on a canvas. This node leverages color mapping and geometric calculations to produce a wide range of artistic and design-oriented outputs.
    - Inputs:
        - `mode` (Required): The 'mode' parameter determines the pattern style of the color bars, affecting the overall appearance of the generated image. Type should be `COMBO[STRING]`.
        - `width` (Required): Specifies the width of the canvas for the color bars, influencing the size of the generated pattern. Type should be `INT`.
        - `height` (Required): Determines the height of the canvas, impacting the vertical scale of the color bar pattern. Type should be `INT`.
        - `color_1` (Required): Specifies the first color used in the color bar pattern, contributing to the visual appeal and theme of the output. Type should be `COMBO[STRING]`.
        - `color_2` (Required): Specifies the second color used in the color bar pattern, complementing the first color to enhance the pattern's aesthetic. Type should be `COMBO[STRING]`.
        - `orientation` (Required): Sets the orientation of the color bars (e.g., vertical, horizontal, diagonal), affecting the direction in which the bars are drawn. Type should be `COMBO[STRING]`.
        - `bar_frequency` (Required): Controls the frequency of the color bars, influencing the density and spacing of the bars in the pattern. Type should be `INT`.
        - `offset` (Required): Determines the starting point of the color bars, allowing for adjustments in the pattern's alignment and positioning. Type should be `FLOAT`.
        - `color1_hex` (Optional): Provides a custom hexadecimal color value for the first color, offering enhanced customization for the color scheme. Type should be `STRING`.
        - `color2_hex` (Optional): Provides a custom hexadecimal color value for the second color, allowing for precise color matching in the pattern. Type should be `STRING`.
    - Outputs:
        - `IMAGE`: The generated image data encapsulating the color bar pattern as per the specified parameters. Type should be `IMAGE`.
        - `show_help`: Provides help or guidance related to the usage of the node and its parameters. Type should be `STRING`.
