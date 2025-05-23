- `CSV Generator [Dream]`: The CSV Generator node is designed for creating and appending data to a CSV file, specifically tailored for animation curve data. It initializes or updates a CSV file with frame and value data, supporting custom CSV dialects for flexible file formatting.
    - Inputs:
        - `frame_counter` (Required): Tracks the current frame in the animation, ensuring accurate timing data is recorded alongside values in the CSV. Type should be `FRAME_COUNTER`.
        - `value` (Required): Specifies the numerical value to be recorded in the CSV file, playing a crucial role in the animation curve's data points. Type should be `FLOAT`.
        - `csvfile` (Required): The path to the CSV file to be created or updated, serving as the primary storage for the animation curve data. Type should be `STRING`.
        - `csv_dialect` (Required): Defines the formatting rules for the CSV file, allowing customization of the file's structure and syntax. Type should be `COMBO[STRING]`.
    - Outputs:
