- `ConditioningAverage`: The ConditioningAverage node is designed to blend conditioning vectors from two sources by averaging them, with the ability to adjust the strength of influence from each source. This functionality is crucial for scenarios where a balanced integration of conditioning information is needed to guide the generation process or modify existing conditioning in a controlled manner.
    - Inputs:
        - `conditioning_to` (Required): Represents the target conditioning vectors to which the blending will be applied. It plays a crucial role in determining the final output by receiving modifications based on the averaged input. Type should be `CONDITIONING`.
        - `conditioning_from` (Required): Serves as the source of conditioning vectors that will be averaged with the target vectors. Its content significantly influences the blending process by providing the base vectors for modification. Type should be `CONDITIONING`.
        - `conditioning_to_strength` (Required): Determines the weighting of the target conditioning vectors in the averaging process, thereby controlling the influence of the source vectors on the final outcome. Type should be `FLOAT`.
    - Outputs:
        - `conditioning`: The output is a modified list of conditioning vectors, reflecting the averaged blend of the input sources with adjusted strengths. Type should be `CONDITIONING`.
