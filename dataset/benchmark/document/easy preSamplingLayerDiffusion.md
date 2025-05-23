- `easy preSamplingLayerDiffusion`: This node is designed to apply a layer diffusion process to images or latent spaces before sampling in a generative model pipeline. It adjusts the blending and diffusion of foreground and background elements based on specified methods and weights, enhancing the control over the generation process. This node is part of the EasyUse/PreSampling category, focusing on preprocessing steps that modify the input data to achieve desired visual effects or characteristics in the final output.
    - Inputs:
        - `pipe` (Required): Represents the pipeline configuration, including settings and states that affect the overall sampling and generation process. Type should be `PIPE_LINE`.
        - `method` (Required): Specifies the method used for layer diffusion, determining how foreground and background elements are blended or diffused. Type should be `COMBO[STRING]`.
        - `weight` (Required): Controls the intensity of the diffusion effect, allowing for fine-tuning of the blend between elements. Type should be `FLOAT`.
        - `steps` (Required): Defines the number of steps to execute in the diffusion process, affecting the depth of the effect. Type should be `INT`.
        - `cfg` (Required): Configuration settings that influence the sampling behavior, such as noise levels and model parameters. Type should be `FLOAT`.
        - `sampler_name` (Required): The name of the sampler to be used, influencing the pattern and quality of the generated output. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): Determines the scheduling algorithm for the sampling process, affecting the progression of the generation. Type should be `COMBO[STRING]`.
        - `denoise` (Required): Adjusts the level of denoising applied during the diffusion process, impacting the clarity and detail of the output. Type should be `FLOAT`.
        - `seed` (Required): A seed value for random number generation, ensuring reproducibility of the results. Type should be `INT`.
        - `image` (Optional): An optional image input for the diffusion process, allowing for direct manipulation of existing images. Type should be `IMAGE`.
        - `blended_image` (Optional): An optional input for providing a pre-blended image, used in conjunction with specific diffusion methods. Type should be `IMAGE`.
        - `mask` (Optional): An optional mask input, used to define areas of the image that should be treated differently during the diffusion process. Type should be `MASK`.
    - Outputs:
        - `pipe`: The modified pipeline configuration after applying the layer diffusion process, ready for further processing or generation steps. Type should be `PIPE_LINE`.
