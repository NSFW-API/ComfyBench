- `VerbingStyler`: The VerbingStyler node dynamically applies stylistic transformations to text inputs based on a selection of styling options defined in its menus. It allows for the customization of text prompts through the application of predefined templates, enhancing the expressiveness and thematic depth of the generated content.
    - Inputs:
        - `text_positive` (Required): The positive text to be styled, serving as the base content for stylistic transformations, impacting the overall thematic presentation of the output. Type should be `STRING`.
        - `text_negative` (Required): The negative text to be styled, transformed according to the selected styling options, altering its thematic and expressive qualities. Type should be `STRING`.
        - `verbing` (Required): Specifies the styling options to be applied, allowing for customization and thematic adjustments to the text. Type should be `COMBO[STRING]`.
        - `log_prompt` (Required): A boolean flag that, when enabled, logs the styling selections and the before-and-after states of the text, aiding in debugging and refinement. Type should be `BOOLEAN`.
    - Outputs:
        - `text_positive`: The styled positive text, reflecting the applied stylistic transformations. Type should be `STRING`.
        - `text_negative`: The fully styled negative text, encapsulating the thematic and expressive alterations resulting from the styling process. Type should be `STRING`.
