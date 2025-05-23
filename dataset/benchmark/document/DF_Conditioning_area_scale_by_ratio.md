- `DF_Conditioning_area_scale_by_ratio`: This node is designed to adjust the scale of conditioning areas based on a specified ratio, modifying both the dimensions and strength of the conditioning to achieve a desired effect. It allows for dynamic resizing of conditioning areas, making it suitable for applications requiring precise control over the conditioning's influence on generated content.
    - Inputs:
        - `conditioning` (Required): The conditioning input represents the current state of conditioning areas that will be scaled. It is crucial for determining the base dimensions and strength that will be modified. Type should be `CONDITIONING`.
        - `modifier` (Required): The modifier is a scaling factor that directly influences the size of the conditioning areas, allowing for proportional resizing. Type should be `FLOAT`.
        - `strength_modifier` (Required): This parameter adjusts the strength of the conditioning, enabling fine-tuning of its impact on the generation process. Type should be `FLOAT`.
    - Outputs:
        - `conditioning`: Returns the conditioning with adjusted area sizes and strength, reflecting the applied scaling modifications. Type should be `CONDITIONING`.
