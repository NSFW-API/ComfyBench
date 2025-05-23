- `Clothing_StateStylerAdvanced`: The Clothing_StateStylerAdvanced node dynamically subclasses from SDXLPromptStylerAdvanced to provide advanced styling capabilities for text prompts related to clothing state. It utilizes a predefined set of templates to modify and enhance input text prompts based on user-selected styling options, aiming to refine the generation of text descriptions or commands in the context of clothing appearance or condition.
    - Inputs:
        - `text_positive_g` (Required): The global positive aspect of the text prompt to be styled, focusing on broader, desirable attributes or outcomes. It's essential for defining the overall positive theme of the styled prompt. Type should be `STRING`.
        - `text_positive_l` (Required): The local positive aspect of the text prompt to be styled, focusing on specific, desirable attributes or outcomes. It complements the global positive prompt by adding detail and nuance. Type should be `STRING`.
        - `text_negative` (Required): The negative aspect of the text prompt to be styled, focusing on undesirable attributes or outcomes. It significantly influences the styling process by identifying elements to be downplayed or avoided in the final prompt. Type should be `STRING`.
        - `clothing_state` (Required): unknown Type should be `COMBO[STRING]`.
        - `negative_prompt_to` (Required): Specifies the scope of the negative styling, whether it applies globally, locally, or both, thus directing how the negative aspects are integrated into the styled prompts. Type should be `COMBO[STRING]`.
        - `log_prompt` (Required): A boolean flag indicating whether to log the original and styled prompts for debugging or review purposes. It helps in understanding the effect of styling choices on the text. Type should be `BOOLEAN`.
    - Outputs:
        - `text_positive_g`: The styled version of the global positive text prompt, enhanced according to the selected styling options. Type should be `STRING`.
        - `text_positive_l`: The styled version of the local positive text prompt, further refined based on styling selections. Type should be `STRING`.
        - `text_positive`: unknown Type should be `STRING`.
        - `text_negative_g`: The styled version of the global negative text prompt, modified to reflect the chosen styling adjustments on a broader scale. Type should be `STRING`.
        - `text_negative_l`: The styled version of the local negative text prompt, adjusted to incorporate the selected styling effects on a more detailed level. Type should be `STRING`.
        - `text_negative`: The combined styled version of the negative text prompts, integrating both global and local adjustments. Type should be `STRING`.
