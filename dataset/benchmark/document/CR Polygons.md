- `CR Polygons`: The CR Polygons node is designed for creating and manipulating polygon shapes within a graphical environment, leveraging the capabilities of matplotlib for visualization. This node facilitates the generation, customization, and rendering of polygonal figures, supporting a wide range of graphical applications.
    - Inputs:
        - `mode` (Required): Specifies the mode of polygon generation, affecting the shape and arrangement of polygons. Type should be `COMBO[STRING]`.
        - `width` (Required): Specifies the width of the canvas on which polygons will be drawn. This parameter influences the overall size of the graphical output. Type should be `INT`.
        - `height` (Required): Determines the height of the canvas, affecting the vertical dimension of the generated polygonal graphics. Type should be `INT`.
        - `rows` (Required): Sets the number of rows in the grid layout for placing polygons. This affects the distribution and organization of polygons on the canvas. Type should be `INT`.
        - `columns` (Required): Defines the number of columns in the grid layout, impacting how polygons are arranged horizontally. Type should be `INT`.
        - `face_color` (Required): Primary color used for filling polygons, contributing to the visual style of the output. Type should be `COMBO[STRING]`.
        - `background_color` (Required): Color used for the canvas background, setting the visual context for the polygons. Type should be `COMBO[STRING]`.
        - `line_color` (Required): Color used for the polygon outlines, enhancing the definition and contrast of shapes. Type should be `COMBO[STRING]`.
        - `line_width` (Required): Specifies the thickness of the polygon outlines, affecting the visual prominence of the shapes. Type should be `INT`.
        - `face_color_hex` (Optional): Hexadecimal representation of the primary color, offering an alternative method for specifying color. Type should be `STRING`.
        - `bg_color_hex` (Optional): Hexadecimal code for the background color, providing precision in color selection. Type should be `STRING`.
        - `line_color_hex` (Optional): Hexadecimal code for the outline color, allowing for exact color matching. Type should be `STRING`.
    - Outputs:
        - `IMAGE`: Generates an image file containing the rendered polygons, serving as the visual output of the node. Type should be `IMAGE`.
        - `show_help`: Provides textual help or guidance related to the node's functionality and usage. Type should be `STRING`.
