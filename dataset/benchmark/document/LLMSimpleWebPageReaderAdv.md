- `LLMSimpleWebPageReaderAdv`: The LLMSimpleWebPageReaderAdv node is designed to fetch and process web pages from a list of URLs, converting them into a structured document format. It optionally converts HTML content to text, facilitating the extraction of readable content from web pages for further analysis or processing.
    - Inputs:
        - `urls` (Required): A list of URLs from which web pages will be fetched. This parameter is essential for the node's operation as it defines the sources of the web content to be processed. Type should be `LIST`.
        - `html_to_text` (Optional): A boolean flag indicating whether the HTML content should be converted to plain text. This affects how the web page content is processed and presented in the output document. Type should be `BOOLEAN`.
    - Outputs:
        - `documents`: The processed web page content, structured as documents. This output is useful for downstream text analysis or content management tasks. Type should be `DOCUMENT`.
