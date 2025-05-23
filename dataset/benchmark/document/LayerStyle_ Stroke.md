- `LayerStyle_ Stroke`: The `LayerStyle: Stroke` node is designed to apply a stroke effect to a layer image, allowing for customization of the stroke's appearance through various parameters such as color, width, and opacity. It operates by blending the layer image with a background image, optionally using a mask for more precise control over the effect's application.
    - Inputs:
        - `background_image` (Required): The background image over which the layer image will be placed. It serves as the canvas for the stroke effect. Type should be `IMAGE`.
        - `layer_image` (Required): The layer image to which the stroke effect will be applied. This image is blended with the background based on the specified parameters. Type should be `IMAGE`.
        - `invert_mask` (Required): A boolean parameter that determines whether the mask applied to the layer image should be inverted, affecting the areas where the stroke effect is applied. Type should be `BOOLEAN`.
        - `blend_mode` (Required): Defines the blending mode used to combine the layer image with the background, influencing the visual outcome of the stroke effect. Type should be `COMBO[STRING]`.
        - `opacity` (Required): The opacity level of the stroke effect, allowing for adjustments in transparency from fully opaque to fully transparent. Type should be `INT`.
        - `stroke_grow` (Required): The amount by which the stroke effect grows or shrinks the layer image, enabling the adjustment of the effect's spread. Type should be `INT`.
        - `stroke_width` (Required): Specifies the width of the stroke effect, determining how thick or thin the stroke appears around the layer image. Type should be `INT`.
        - `blur` (Required): The level of blur applied to the stroke effect, which can soften the edges of the stroke for a more subtle appearance. Type should be `INT`.
        - `stroke_color` (Required): The color of the stroke effect, allowing for customization of the stroke's appearance to match the desired aesthetic. Type should be `STRING`.
        - `layer_mask` (Optional): An optional mask that can be applied to the layer image, providing more control over where the stroke effect is applied. Type should be `MASK`.
    - Outputs:
        - `image`: The resulting image after applying the stroke effect, combining the layer and background images with the specified stroke parameters. Type should be `IMAGE`.
