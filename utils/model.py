import os
import json
import yaml

from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

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
        retriever = self.storage.as_retriever(search_kwargs={"k": count})
        reference_list = retriever.invoke(query)
        return reference_list


def invoke_completion(message: str) -> tuple[str, any]:
    client = OpenAI(
        base_url=COMPLETION_BASE_URL,
        api_key=COMPLETION_API_KEY
    )

    try:
        # Using the new Responses API
        response = client.responses.create(
            model=COMPLETION_MODEL_NAME,
            input=message,  # Simply pass the message as input
            tools=[{"type": "web_search"}],
            tool_choice={"type":"web_search"},
            **COMPLETION_HYPER_PARAMETER
        )
        answer = response.output_text  # Use output_text instead of choices[0].message.content
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage


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
