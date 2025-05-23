- `Laboratory [Dream]`: The Laboratory node is designed to generate and manipulate numerical values based on a variety of modes such as random uniform, random bell, ladder, and random walk. It allows for dynamic value generation within specified ranges, making it a versatile tool for experiments and simulations in creative projects.
    - Inputs:
        - `frame_counter` (Required): A counter tracking the current frame within a sequence, essential for determining when to renew the generated value based on the specified policy. Type should be `FRAME_COUNTER`.
        - `key` (Required): A unique identifier for the generated value, supporting default randomization to ensure uniqueness. It plays a crucial role in value tracking and retrieval across frames. Type should be `STRING`.
        - `seed` (Required): Determines the starting point of the random number generation, enabling reproducible results. It's essential for consistency in simulations or when re-generating values. Type should be `INT`.
        - `renew_policy` (Required): Defines the policy for renewing the generated value, affecting how often new values are produced based on frame changes or initial generation. Type should be `COMBO[STRING]`.
        - `min_value` (Required): Sets the lower bound for the generated value, crucial for defining the value range and ensuring outputs fall within expected limits. Type should be `FLOAT`.
        - `max_value` (Required): Establishes the upper limit for the generated value, essential for controlling the range and precision of the output. Type should be `FLOAT`.
        - `mode` (Required): Specifies the method of value generation, influencing the distribution and variation of the output values. Type should be `COMBO[STRING]`.
        - `step_size` (Optional): Determines the increment size between values in certain modes, impacting the granularity and smoothness of value transitions. Type should be `FLOAT`.
    - Outputs:
        - `FLOAT`: The primary generated float value, serving as the versatile output for various applications. Type should be `FLOAT`.
        - `INT`: An integer representation of the primary generated value, offering a discrete alternative for specific needs. Type should be `INT`.
        - `log_entry`: A log entry detailing the generation or reuse of the value, providing insights and traceability for debugging and analysis. Type should be `LOG_ENTRY`.
