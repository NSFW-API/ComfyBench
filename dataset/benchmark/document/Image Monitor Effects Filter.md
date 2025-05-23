- `Image Monitor Effects Filter`: This node applies various monitor effect filters to an image, simulating digital, signal, and TV distortions. It allows for customization of the distortion intensity and offset, providing a versatile tool for creating visually unique images.
    - Inputs:
        - `image` (Required): The input image to which the monitor effect filters will be applied. It serves as the base for the distortion effects. Type should be `IMAGE`.
        - `mode` (Required): Specifies the type of distortion effect to apply: Digital Distortion, Signal Distortion, or TV Distortion. This choice determines the visual style of the output image. Type should be `COMBO[STRING]`.
        - `amplitude` (Required): Controls the intensity of the distortion effect. A higher value results in more pronounced distortions. Type should be `INT`.
        - `offset` (Required): Adjusts the offset of the distortion effect, allowing for further customization of the visual outcome. Type should be `INT`.
    - Outputs:
        - `image`: The output image after applying the selected monitor effect filter. It showcases the visual distortions as specified by the input parameters. Type should be `IMAGE`.
