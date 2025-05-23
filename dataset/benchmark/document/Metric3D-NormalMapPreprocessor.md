- `Metric3D-NormalMapPreprocessor`: This node preprocesses images for normal map estimation using a 3D metric model. It leverages a pre-trained Metric3DDetector model, configurable with different backbone architectures and intrinsic camera parameters, to compute normal maps from input images. The node aims to enhance depth and normal estimation tasks by providing detailed normal maps, which are crucial for accurate 3D reconstruction and analysis.
    - Inputs:
        - `image` (Required): The input image to be processed for normal map estimation. Type should be `IMAGE`.
        - `backbone` (Optional): Specifies the backbone architecture for the Metric3DDetector model. Different backbones offer varying levels of detail and accuracy in the normal map estimation. Type should be `COMBO[STRING]`.
        - `fx` (Optional): The focal length of the camera in the x-axis, used to calibrate the Metric3DDetector model for accurate normal map estimation. Type should be `INT`.
        - `fy` (Optional): The focal length of the camera in the y-axis, essential for calibrating the model to accurately estimate normal maps. Type should be `INT`.
        - `resolution` (Optional): The resolution to which the input image is scaled before processing. This parameter can affect the detail level of the estimated normal map. Type should be `INT`.
    - Outputs:
        - `image`: The output is an image representing the estimated normal map, which visualizes the orientation of surfaces in the scene. Type should be `IMAGE`.
