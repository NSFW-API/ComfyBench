- `Fooocus_KSampler`: The Fooocus_KSampler node enhances the sampling process in art generation by introducing an adjustable sharpness parameter to the traditional KSampler functionality. This allows for more precise control over the sharpness of generated images, catering to the specific needs of art ventures.
    - Inputs:
        - `model` (Required): Specifies the model to be used for the sampling process, serving as the foundation for generating images. Type should be `MODEL`.
        - `seed` (Required): Determines the initial seed for randomness in the sampling process, ensuring reproducibility of results. Type should be `INT`.
        - `steps` (Required): Defines the number of steps to be taken in the sampling process, affecting the detail and quality of the generated images. Type should be `INT`.
        - `cfg` (Required): Controls the conditioning-free guidance scale, influencing the adherence to specified conditions. Type should be `FLOAT`.
        - `sampler_name` (Required): Selects the specific sampler algorithm to be used, impacting the sampling behavior and output quality. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): Chooses the scheduler for controlling the sampling process, affecting the progression of image generation. Type should be `COMBO[STRING]`.
        - `positive` (Required): Sets positive conditioning to guide the sampling towards desired attributes. Type should be `CONDITIONING`.
        - `negative` (Required): Applies negative conditioning to steer the sampling away from undesired attributes. Type should be `CONDITIONING`.
        - `latent_image` (Required): Provides an initial latent image to be refined or altered through the sampling process. Type should be `LATENT`.
        - `denoise` (Required): Adjusts the level of denoising applied to the generated images, affecting clarity and detail. Type should be `FLOAT`.
        - `sharpness` (Optional): The sharpness parameter allows users to adjust the sharpness level of the generated images, providing a means to fine-tune the visual output according to artistic preferences. Type should be `FLOAT`.
    - Outputs:
        - `latent`: The output latent representation of the generated image, ready for further processing or conversion to a visual format. Type should be `LATENT`.
