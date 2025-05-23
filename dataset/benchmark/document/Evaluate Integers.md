- `Evaluate Integers`: The node 'Evaluate Integers' dynamically evaluates mathematical expressions involving integers, using variables 'a', 'b', and 'c' as inputs. It supports basic arithmetic operations and returns the result in multiple formats (integer, float, and string), optionally printing the outcome to the console.
    - Inputs:
        - `python_expression` (Required): Specifies the mathematical expression to be evaluated. It can include variables 'a', 'b', and 'c' and supports basic arithmetic operations. Type should be `STRING`.
        - `print_to_console` (Required): Controls whether the evaluation result is printed to the console. Useful for debugging or direct output visualization. Type should be `COMBO[STRING]`.
        - `a` (Optional): Represents the first integer variable that can be used in the python_expression. Type should be `INT`.
        - `b` (Optional): Represents the second integer variable that can be used in the python_expression. Type should be `INT`.
        - `c` (Optional): Represents the third integer variable that can be used in the python_expression. Type should be `INT`.
    - Outputs:
        - `int`: The evaluated result of the python_expression as an integer. Type should be `INT`.
        - `float`: The evaluated result of the python_expression as a float. Type should be `FLOAT`.
        - `string`: The evaluated result of the python_expression as a string. Type should be `STRING`.
