- `easy hunyuanDiTLoader`: This node is designed to load and configure the HunYuanDiT model for image generation tasks. It prepares the model by loading the necessary checkpoints, VAE, and CLIP components, and sets up the environment for generating images based on text prompts. The node facilitates the integration of the HunYuanDiT model into the workflow, ensuring that all components are correctly initialized and ready for use.
    - Inputs:
        - `ckpt_name` (Required): Specifies the checkpoint name for loading the model, essential for initializing the HunYuanDiT model with the correct weights and configurations. Type should be `COMBO[STRING]`.
        - `model_name` (Required): Defines the specific model configuration to use, allowing for customization and optimization of the image generation process. Type should be `COMBO[STRING]`.
        - `vae_name` (Required): Indicates the VAE component to load, crucial for the image encoding and decoding processes within the HunYuanDiT model. Type should be `COMBO[STRING]`.
        - `clip_name` (Required): Determines the CLIP model to integrate, used for processing text prompts and guiding the image generation. Type should be `COMBO[STRING]`.
        - `mt5_name` (Required): Specifies the MT5 model for additional text processing capabilities, enhancing the model's understanding of text prompts. Type should be `COMBO[STRING]`.
        - `device` (Required): Selects the computation device (e.g., CPU, GPU) for model execution, affecting performance and efficiency. Type should be `COMBO[STRING]`.
        - `dtype` (Required): Sets the data type for model operations, ensuring compatibility and optimal performance. Type should be `COMBO[STRING]`.
        - `resolution` (Required): Defines the resolution of the generated images, allowing for control over image quality and detail. Type should be `COMBO[STRING]`.
        - `empty_latent_width` (Required): Specifies the width of the empty latent space, important for initializing the image generation process. Type should be `INT`.
        - `empty_latent_height` (Required): Specifies the height of the empty latent space, important for initializing the image generation process. Type should be `INT`.
        - `positive` (Required): Contains positive text prompts that guide the image generation towards desired attributes. Type should be `STRING`.
        - `negative` (Required): Contains negative text prompts that guide the image generation away from undesired attributes. Type should be `STRING`.
        - `batch_size` (Required): Determines the number of images to generate in a single batch, impacting performance and output volume. Type should be `INT`.
    - Outputs:
        - `pipe`: Returns the configured pipeline for image generation, including all necessary models and settings. Type should be `PIPE_LINE`.
        - `model`: Provides the loaded and configured model ready for image generation tasks. Type should be `MODEL`.
        - `vae`: Returns the loaded VAE component, essential for the image encoding and decoding processes. Type should be `VAE`.
