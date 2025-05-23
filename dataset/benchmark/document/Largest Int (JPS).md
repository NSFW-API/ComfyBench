- `Largest Int (JPS)`: The node determines the largest and smallest integers between two given values, and also indicates if the first integer is larger than the second. It abstracts the comparison logic into a simple interface for users.
    - Inputs:
        - `int_a` (Required): Represents the first integer to be compared. Its value influences the determination of the larger and smaller integers, as well as the flag indicating if it is larger than the second integer. Type should be `INT`.
        - `int_b` (Required): Represents the second integer to be compared. It is used alongside the first integer to determine the larger and smaller values, and to set the flag indicating if the first integer is larger. Type should be `INT`.
    - Outputs:
        - `larger_int`: The larger integer between the two inputs. Type should be `INT`.
        - `smaller_int`: The smaller integer between the two inputs. Type should be `INT`.
        - `is_a_larger`: A flag indicating if the first input integer is larger than the second. Type should be `INT`.
