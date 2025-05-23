- `LayerFilter_ ChannelShake`: The ChannelShake node applies a unique visual effect to images by shifting the color channels (RGB) in specified directions and distances. This manipulation creates a shaking or displacement effect on the image, offering a creative way to alter the visual appearance based on angle and distance parameters.
    - Inputs:
        - `image` (Required): The input image to apply the channel shake effect. This parameter is crucial for defining the base image on which the effect will be applied. Type should be `IMAGE`.
        - `distance` (Required): Specifies the distance of the channel shift. This parameter influences the intensity of the shaking effect, with larger values resulting in more pronounced shifts. Type should be `INT`.
        - `angle` (Required): Determines the angle at which the channels will be shifted. This affects the direction of the shake effect, allowing for a wide range of visual distortions. Type should be `FLOAT`.
        - `mode` (Required): Defines the order in which the RGB channels are shifted. This selection alters the visual outcome of the effect, offering various combinations for creative experimentation. Type should be `COMBO[STRING]`.
    - Outputs:
        - `image`: The output image after applying the channel shake effect. This image showcases the visual transformation achieved through the specified channel shifts. Type should be `IMAGE`.
