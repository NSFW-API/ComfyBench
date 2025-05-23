- `CR Color Gradient`: The CR Color Gradient node is designed to generate a color gradient between two specified colors over a given canvas. It allows for customization of the gradient's orientation, transition style, and distance, enabling the creation of visually appealing backgrounds or elements with smooth color transitions.
    - Inputs:
        - `width` (Required): The width of the canvas on which the gradient will be applied, impacting the horizontal dimension of the gradient. Type should be `INT`.
        - `height` (Required): The height of the canvas on which the gradient will be applied, affecting the vertical dimension of the gradient. Type should be `INT`.
        - `start_color` (Required): Specifies the starting color of the gradient. It can be a predefined color name or 'custom' to use a specific hex color code, influencing the gradient's initial color. Type should be `COMBO[STRING]`.
        - `end_color` (Required): Defines the ending color of the gradient. Similar to 'start_color', it can be a predefined color name or 'custom' for a hex color code, determining the gradient's final color. Type should be `COMBO[STRING]`.
        - `gradient_distance` (Required): Specifies the distance over which the gradient transition occurs, with a value between 0 and 1 representing the proportion of the canvas covered by the gradient. Type should be `FLOAT`.
        - `linear_transition` (Required): A value between 0 and 1 indicating the position of the gradient transition's center point, with 0 being the start and 1 the end of the canvas. Type should be `FLOAT`.
        - `orientation` (Required): Determines the orientation of the gradient (horizontal or vertical), influencing the direction in which the color transition occurs. Type should be `COMBO[STRING]`.
        - `start_color_hex` (Optional): The hex color code for the starting color, used when 'start_color' is set to 'custom'. It provides precise control over the gradient's initial color. Type should be `STRING`.
        - `end_color_hex` (Optional): The hex color code for the ending color, used when 'end_color' is set to 'custom'. It allows for exact specification of the gradient's final color. Type should be `STRING`.
    - Outputs:
        - `IMAGE`: The resulting image with the applied color gradient, ready for display or further processing. Type should be `IMAGE`.
        - `show_help`: A URL to a help page providing additional information and guidance on using the CR Color Gradient node. Type should be `STRING`.
