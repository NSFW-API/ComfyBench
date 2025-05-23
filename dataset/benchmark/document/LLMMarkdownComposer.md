- `LLMMarkdownComposer`: The LLMMarkdownComposer node is designed to transform input text into well-structured Markdown documents. It leverages a language model to interpret and format the given text according to specified classifiers and additional directions, ensuring the output adheres to Markdown syntax while incorporating all provided data.
    - Inputs:
        - `llm_model` (Required): Specifies the language model to use for generating the Markdown document. It plays a crucial role in understanding the input text and formatting it according to Markdown syntax. Type should be `LLM_MODEL`.
        - `text_input` (Required): The primary text input that will be transformed into a Markdown document. This text serves as the base content for the Markdown generation process. Type should be `STRING`.
        - `classifier_list` (Required): A comma-separated list of classifiers that guide the language model in structuring the Markdown document. These classifiers help in categorizing and formatting the input text appropriately. Type should be `STRING`.
        - `extra_directions` (Optional): Additional instructions for the language model to follow when generating the Markdown document. These directions can include specific formatting requests or content structuring guidelines. Type should be `STRING`.
    - Outputs:
        - `markdown_output`: The generated Markdown document, structured and formatted according to the input specifications and additional directions. Type should be `STRING`.
