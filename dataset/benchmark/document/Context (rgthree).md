- `Context (rgthree)`: The Context (rgthree) node serves as the foundational context node, designed to be highly compatible with 1.5 applications and other context nodes. It focuses on converting input context parameters into a structured output that is optimized for most use cases, maintaining both forward and backward compatibility.
    - Inputs:
        - `base_ctx` (Optional): The base context to be converted or enhanced. It serves as the starting point for the conversion process, allowing for the integration or modification of additional context parameters. Type should be `RGTHREE_CONTEXT`.
        - `model` (Optional): Specifies the model to be used in the context, allowing for customization and flexibility in processing. Type should be `MODEL`.
        - `clip` (Optional): Defines the CLIP model to be incorporated into the context, enhancing the processing capabilities. Type should be `CLIP`.
        - `vae` (Optional): Indicates the VAE model to be included in the context, contributing to the generation process. Type should be `VAE`.
        - `positive` (Optional): A positive conditioning factor to guide the generation towards desired outcomes. Type should be `CONDITIONING`.
        - `negative` (Optional): A negative conditioning factor to steer the generation away from undesired outcomes. Type should be `CONDITIONING`.
        - `latent` (Optional): Specifies the latent space representation to be used in the context, enabling advanced manipulation. Type should be `LATENT`.
        - `images` (Optional): Defines the images to be included in the context, allowing for visual data integration. Type should be `IMAGE`.
        - `seed` (Optional): Sets the seed for random number generation, ensuring reproducibility of results. Type should be `INT`.
    - Outputs:
        - `CONTEXT`: The structured output context optimized for use in various applications. Type should be `RGTHREE_CONTEXT`.
        - `MODEL`: The model used within the context, reflecting the specified input. Type should be `MODEL`.
        - `CLIP`: The CLIP model incorporated into the context, as specified in the input. Type should be `CLIP`.
        - `VAE`: The VAE model included in the context, as per the input parameters. Type should be `VAE`.
        - `POSITIVE`: The positive conditioning factor applied in the context, guiding the generation process. Type should be `CONDITIONING`.
        - `NEGATIVE`: The negative conditioning factor used in the context to avoid undesired outcomes. Type should be `CONDITIONING`.
        - `LATENT`: The latent space representation utilized in the context for advanced manipulation. Type should be `LATENT`.
        - `IMAGE`: The images included in the context, integrating visual data. Type should be `IMAGE`.
        - `SEED`: The seed used for random number generation within the context, ensuring consistency. Type should be `INT`.
