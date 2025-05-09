import json
import os

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from utils.parser import parse_prompt_to_code, parse_prompt_to_markdown

with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    if config['proxy']['http_proxy'] is not None:
        os.environ['http_proxy'] = config['proxy']['http_proxy']
    if config['proxy']['https_proxy'] is not None:
        os.environ['https_proxy'] = config['proxy']['https_proxy']
    EMBEDDING_BASE_URL = config['embedding']['base_url']
    EMBEDDING_API_KEY = config['embedding']['api_key']
    EMBEDDING_MODEL_NAME = config['embedding']['model_name']
    COMPLETION_BASE_URL = config['completion']['base_url']
    COMPLETION_API_KEY = config['completion']['api_key']
    COMPLETION_MODEL_NAME = config['completion']['model_name']
    COMPLETION_HYPER_PARAMETER = config['completion']['hyper_parameter']
    VISION_BASE_URL = config['vision']['base_url']
    VISION_API_KEY = config['vision']['api_key']
    VISION_MODEL_NAME = config['vision']['model_name']
    VISION_HYPER_PARAMETER = config['vision']['hyper_parameter']
    # Load web search configuration
    WEB_SEARCH_ENABLED = config.get('web_search', {}).get('enabled', True)
    WEB_SEARCH_MAX_RESULTS = config.get('web_search', {}).get('max_results', 3)
    WEB_SEARCH_TOKEN_BUDGET = config.get('web_search', {}).get('token_budget', 512)


class ReferenceStorage(object):
    def __init__(self, cache: str = './cache/reference_storage'):
        self.embedding = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY
        )

        with open('./dataset/benchmark/workflow/meta.json') as file:
            meta = json.load(file)
        self.document_list = []

        for index, information in meta.items():
            with open(f'./dataset/benchmark/workflow/{index}.json') as file:
                prompt = json.load(file)
            information['prompt'] = json.dumps(prompt)
            information['code'] = parse_prompt_to_code(prompt)
            information['markdown'] = parse_prompt_to_markdown(prompt)
            content = f'{information["name"]}: {information["function"]} {information["principle"]}'
            self.document_list.append(Document(content, metadata=information))

        if os.path.exists(cache):
            self.storage = Chroma(
                embedding_function=self.embedding,
                persist_directory=cache
            )
        else:
            self.storage = Chroma.from_documents(
                self.document_list,
                embedding=self.embedding,
                persist_directory=cache
            )

    def retrieve(self, query: str, count: int = 5) -> list[Document]:
        # Retrieve from local vector database
        retriever = self.storage.as_retriever(search_kwargs={"k": count})
        reference_list = retriever.invoke(query)

        # Perform web search if enabled
        if WEB_SEARCH_ENABLED:
            try:
                # Create search query based on ComfyUI context
                search_query = f"ComfyUI {query}"

                # Get web search results
                web_docs = self._search_web(search_query)

                # Append web documents to reference list
                reference_list.extend(web_docs)

                # Optionally, upsert web results to Chroma for future use
                # self._upsert_web_snippets(web_docs)

            except Exception as e:
                print(f"Web search error: {e}")
                # Continue with local references only

        return reference_list

    def _search_web(self, query: str) -> list[Document]:
        """Perform web search and convert results to Document objects."""
        web_docs = []

        try:
            # Use the invoke_completion function with web_search tool enabled
            search_prompt = f"Find information about ComfyUI nodes and workflows related to: {query}. Focus on documentation, examples, and usage patterns."
            results, _ = invoke_completion(search_prompt)

            if results and not results.startswith('Error:'):
                # Create a Document with the web search results
                doc = Document(
                    page_content=results[:WEB_SEARCH_TOKEN_BUDGET],  # Truncate to token budget
                    metadata={
                        "name": f"web:comfyui_{query.replace(' ', '_')[:30]}",
                        "source": "web",
                        "code": "",  # Empty unless specific code is extracted
                        "function": f"Web search results for: {query}",
                        "principle": "Contains the latest documentation and examples from the web."
                    }
                )
                web_docs.append(doc)
        except Exception as e:
            print(f"Error in web search: {e}")

        return web_docs[:WEB_SEARCH_MAX_RESULTS]  # Limit number of results

    def _upsert_web_snippets(self, web_docs: list[Document]):
        """Upsert web documents to Chroma for future retrieval."""
        if web_docs:
            try:
                self.storage.add_documents(web_docs)
            except Exception as e:
                print(f"Error upserting web docs to Chroma: {e}")


def invoke_completion(message: str) -> tuple[str, any]:
    client = OpenAI(
        base_url=COMPLETION_BASE_URL,
        api_key=COMPLETION_API_KEY
    )

    try:
        # Using the new Responses API with web_search tool enabled
        response = client.responses.create(
            model=COMPLETION_MODEL_NAME,
            input=message,  # Simply pass the message as input
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"},
            **COMPLETION_HYPER_PARAMETER
        )
        answer = response.output_text  # Use output_text instead of choices[0].message.content
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage


def invoke_llm_with_websearch(prompt: str):
    print(prompt)
    """
    Exactly the old invoke_completion() *minus* response_format.
    Returns raw string so that json_adapter() can post-process.
    """
    client = OpenAI(base_url=COMPLETION_BASE_URL, api_key=COMPLETION_API_KEY)
    try:
        resp = client.responses.create(
            model=COMPLETION_MODEL_NAME,
            input=prompt,
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"},
            # ─────────── NO response_format ───────────
            **COMPLETION_HYPER_PARAMETER,
        )
        print(resp)
        return resp.output_text, resp.usage
    except Exception as e:
        return f"Error: {e}", None


def invoke_vision(message: any) -> tuple[str, any]:
    client = OpenAI(
        base_url=VISION_BASE_URL,
        api_key=VISION_API_KEY
    )

    try:
        # Extract text and images from the message structure
        content = message[0]['content']

        # Extract text input (first text block or empty string)
        text_content = next((item['text'] for item in content if item['type'] == 'text'), "")

        # Extract image URLs
        image_urls = []
        for item in content:
            if item['type'] == 'image_url':
                img_url = item['image_url']['url']
                image_urls.append(img_url)

        # Create the call with the new Responses API
        response = client.responses.create(
            model=VISION_MODEL_NAME,
            input=text_content,
            input_images=image_urls if image_urls else None,
            **VISION_HYPER_PARAMETER
        )
        answer = response.output_text
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage
