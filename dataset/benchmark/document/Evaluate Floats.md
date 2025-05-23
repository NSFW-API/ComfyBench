- `Evaluate Floats`: The `TSC_EvaluateFloats` node is designed to evaluate mathematical expressions involving floats, converting the results into multiple formats (integer, float, and string) for further use. It supports dynamic input through variables and can optionally print the evaluation results to the console for debugging or informational purposes.
    - Inputs:
        - `python_expression` (Required): The mathematical expression to be evaluated, allowing for dynamic computation involving floats. It plays a crucial role in determining the node's output by dictating the operation to be performed. Type should be `STRING`.
        - `print_to_console` (Required): A flag indicating whether the evaluation result should be printed to the console, aiding in debugging or providing immediate visual feedback of the operation's outcome. Type should be `COMBO[STRING]`.
        - `a` (Optional): An optional variable 'a' that can be used within the python_expression for dynamic evaluation. Type should be `FLOAT`.
        - `b` (Optional): An optional variable 'b' that can be used within the python_expression for dynamic evaluation. Type should be `FLOAT`.
        - `c` (Optional): An optional variable 'c' that can be used within the python_expression for dynamic evaluation. Type should be `FLOAT`.
    - Outputs:
        - `int`: The integer representation of the evaluated expression's result. Type should be `INT`.
        - `float`: The float representation of the evaluated expression's result. Type should be `FLOAT`.
        - `string`: The string representation of the evaluated expression's result. Type should be `STRING`.
