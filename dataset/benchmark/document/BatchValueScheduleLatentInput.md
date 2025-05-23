- `BatchValueScheduleLatentInput`: The BatchValueScheduleLatentInput node is designed to process latent inputs in the context of batch scheduling. It focuses on handling and transforming latent data according to a value schedule, enabling dynamic adjustments and manipulations of latent vectors based on specified scheduling parameters.
    - Inputs:
        - `text` (Required): The 'text' parameter is a string that specifies the key frames and their corresponding values for the value schedule. It plays a crucial role in determining how the latent inputs are transformed over time. Type should be `STRING`.
        - `num_latents` (Required): The 'num_latents' parameter represents the latent inputs to be processed. It is essential for defining the latent vectors that will be adjusted according to the value schedule. Type should be `LATENT`.
        - `print_output` (Required): The 'print_output' parameter controls whether the scheduling results are printed. It allows for optional debugging or visualization of the value schedule's effect on the latent inputs. Type should be `BOOLEAN`.
    - Outputs:
        - `float`: This output represents the scheduled values as a floating-point number, reflecting the dynamic adjustments made to the latent inputs. Type should be `FLOAT`.
        - `int`: This output provides the integer representation of the scheduled values, offering an alternative numerical perspective on the adjustments. Type should be `INT`.
        - `latent`: This output includes the transformed latent inputs, showcasing the result of the scheduling process on the latent vectors. Type should be `LATENT`.
