import ast
import json

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from utils.adapter import json_adapter
from utils.parser import parse_code_to_prompt
from utils.model import ReferenceStorage, invoke_completion, invoke_llm_with_websearch


class ComfyAgentState(object):
    def __init__(self, instruction: str, analysis: str, reference_list: list[Document], step_limitation: int):
        self.instruction = instruction
        self.analysis = analysis
        self.reference_list = reference_list
        self.step_limitation = step_limitation
        self.current_step = 0
        self.history_list = []
        self.workspace_dict = {}

    def fetch_reference(self, name: str):
        for reference in self.reference_list:
            if reference.metadata['name'] == name:
                return reference
        return None

    def update_reference(self, reference_list: list):
        self.reference_list = reference_list

    def update_workspace(self, code: str, function: str, principle: str):
        self.workspace_dict['code'] = code
        self.workspace_dict['function'] = function
        self.workspace_dict['principle'] = principle

    def update_history(self, thought: str, plan: str, action: str):
        self.current_step += 1
        self.history_list.append({
            'step': self.current_step,
            'thought': thought,
            'plan': plan,
            'action': action
        })


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


analyzer_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Analysis

Based on the instruction, analyze the core requirements (e.g. object, pose, style, etc.) and the expected paradigm (e.g., text-to-image, image-to-image, image-to-video, etc.), so that we can retrieve relevant workflows for your reference. You should not consider the quality unless specific requirements (e.g. upscaling, interpolation, refinement, etc.) are mentioned. Note that you do not need to provide the workflow. Make sure your analysis is clear and concise within a single paragraph.
'''


def format_analyzer_agent_prompt(instruction: str) -> str:
    instruction_content = instruction
    prompt_content = analyzer_prompt.format(
        instruction=instruction_content.strip()
    )
    return prompt_content


planner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable-Diffusion pipelines so that users can design their own workflows to generate highly-customised artworks.  
You are an expert in ComfyUI helping users to design workflows according to their requirements.

You must create a ComfyUI workflow for the following task:

{instruction}

The core requirements and expected paradigm have been analysed as follows:

{analysis}

---

## Reference  ( includes snippets fetched from the live web )

{reference}

### Context – about those references
* Some items above are **fresh web-search excerpts** and can be partially irrelevant or noisy.  
* **Ignore** text that is clearly off-topic or not useful for building the workflow.

---

## History  (latest entry last)

{history}

---

## Current Workspace

{workspace}

---

## Your job / remaining steps
There are **{limitation} steps** left before the agent must finish.  
At each step decide what to do next.

### Available actions
* `load name=<reference_name>` – replace workspace with a reference workflow  
* `combine name=<reference_name>` – merge a reference workflow into workspace  
* `adapt prompt="<brief change description>"` – tweak parameters / prompts  
* `retrieve prompt="<what you still need>"` – fetch new references  
* `finish` – declare the workflow complete

---

## Output Guidelines  ( free-form )

* **Write naturally** – no XML/JSON tags are required; structured formatting will be added later.  
* A helpful convention (not mandatory) is to prefix sections like:  
Thought: ...
Plan: ...
Action: combine name=text_to_image_basic
The adapter layer will parse whatever you write.

* Keep the **Action** on its own line, starting with the word **Action:** followed by one of the commands above.

* Respect the general rules you already know (e.g. don’t `adapt` twice in a row, don’t `load` unless history is empty, and finish before steps run out).

Now think, plan, and state your next action.
'''

def format_planner_agent_prompt(state: ComfyAgentState) -> str:
    instruction_content = state.instruction
    analysis_content = state.analysis

    reference_content = ''
    for reference in state.reference_list:
        reference_content += f'- Workflow: {reference.metadata["name"]}\n\n'
        reference_content += f'<function>\n{reference.metadata["function"]}\n</function>\n\n'
        reference_content += f'<principle>\n{reference.metadata["principle"]}\n</principle>\n\n'

    history_content = ''
    if len(state.history_list) == 0:
        history_content += '- The history is empty.'
    else:
        for record in state.history_list:
            history_content += f'- Step: {record["step"]}\n\n'
            history_content += f'<thought>\n{record["thought"]}\n</thought>\n\n'
            history_content += f'<plan>\n{record["plan"]}\n</plan>\n\n'
            history_content += f'<action>\n{record["action"]}\n</action>\n\n'

    workspace_content = ''
    if len(state.workspace_dict) == 0:
        workspace_content += '- The workspace is empty.'
    else:
        workspace_content += f'<code>\n{state.workspace_dict["code"]}\n</code>\n\n'
        workspace_content += f'<function>\n{state.workspace_dict["function"]}\n</function>\n\n'
        workspace_content += f'<principle>\n{state.workspace_dict["principle"]}\n</principle>\n\n'

    limitation_content = str(state.step_limitation - state.current_step)

    prompt_content = planner_prompt.format(
        instruction=instruction_content.strip(),
        analysis=analysis_content.strip(),
        reference=reference_content.strip(),
        history=history_content.strip(),
        workspace=workspace_content.strip(),
        limitation=limitation_content.strip()
    )
    return prompt_content


def parse_planner_agent_answer(answer: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    thought = safe_extract_from_soup(soup, 'thought')
    plan = safe_extract_from_soup(soup, 'plan')
    action = safe_extract_from_soup(soup, 'action')
    return thought, plan, action


def parse_planner_agent_action(action: str) -> tuple[str, dict]:
    node = ast.parse(action)
    call = node.body[0].value
    command = call.func.id
    arguments = {}
    for keyword in call.keywords:
        arguments[keyword.arg] = keyword.value.value
    return command, arguments


combiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

The core requirements and the expected paradigm are analyzed as follows:

{analysis}

## Reference

The code and annotation of the current workflow you are referring to are presented as follows:

{reference}

## Workspace

The code and annotation of the current workflow you are working on are presented as follows:

{workspace}

## Combination

Based on the current working progress, your schedule is presented as follows:

{schedule}

You are working on the first step of your schedule. In other words, you should combine the reference workflow with the current workflow according to your schedule.

First, you should provide your Python code to formulate the updated workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. You should also avoid reusing the same variable name, even if the variable is temporary. Your code should be enclosed with "<code>" tag. For example: <code>output = node(input)</code>.

After that, you should provide an annotation as in the reference, including the function and principle of the updated workflow. The function should be enclosed with "<function>" tag. For example: <function>This workflow generates a high-resolution image of a running horse.</function>. The principle should be enclosed with "<principle>" tag. For example: <principle>The workflow first generates a low-resolution image using the text-to-image pipeline and then applies an upscaling module to improve the resolution.</principle>.

Now, provide your code and annotation with the required format.
'''


def format_combiner_agent_prompt(state: ComfyAgentState, reference: Document) -> str:
    instruction_content = state.instruction
    analysis_content = state.analysis

    reference_content = ''
    reference_content += f'<code>\n{reference.metadata["code"]}\n</code>\n\n'
    reference_content += f'<function>\n{reference.metadata["function"]}\n</function>\n\n'
    reference_content += f'<principle>\n{reference.metadata["principle"]}\n</principle>\n\n'

    workspace_content = ''
    if len(state.workspace_dict) == 0:
        workspace_content += '- The workspace is empty.'
    else:
        workspace_content += f'<code>\n{state.workspace_dict["code"]}\n</code>\n\n'
        workspace_content += f'<function>\n{state.workspace_dict["function"]}\n</function>\n\n'
        workspace_content += f'<principle>\n{state.workspace_dict["principle"]}\n</principle>\n\n'

    schedule_content = str(state.history_list[-1]['plan'])

    prompt_text = combiner_prompt.format(
        instruction=instruction_content.strip(),
        analysis=analysis_content.strip(),
        reference=reference_content.strip(),
        workspace=workspace_content.strip(),
        schedule=schedule_content.strip()
    )
    return prompt_text


def parse_combiner_agent_answer(answer: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    function = safe_extract_from_soup(soup, 'function')
    principle = safe_extract_from_soup(soup, 'principle')
    return code, function, principle


adapter_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

The core requirements and the expected paradigm are analyzed as follows:

{analysis}

## Workspace

The code and annotation of the current workflow you are working on are presented as follows:

{workspace}

## Adaptation

Based on the current working progress, your schedule is presented as follows:

{schedule}

You are working on the first step of your schedule. In other words, you should modify the parameters in the current workflow according to your schedule. The adaptation you want to make is specified as follows:

{adaptation}

First, you should provide your Python code to formulate the updated workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. You should also avoid reusing the same variable name, even if the variable is temporary. Your code should be enclosed with "<code>" tag. For example: <code>output = node(input)</code>.

After that, you should provide an annotation as in the reference, including the function and principle of the updated workflow. The function should be enclosed with "<function>" tag. For example: <function>This workflow generates a high-resolution image of a running horse.</function>. The principle should be enclosed with "<principle>" tag. For example: <principle>The workflow first generates a low-resolution image using the text-to-image pipeline and then applies an upscaling module to improve the resolution.</principle>.

Now, provide your code and annotation with the required format.
'''


def format_adapter_agent_prompt(state: ComfyAgentState, adaptation: str) -> str:
    instruction_content = state.instruction
    analysis_content = state.analysis

    workspace_content = ''
    if len(state.workspace_dict) == 0:
        workspace_content += '- The workspace is empty.'
    else:
        workspace_content += f'<code>\n{state.workspace_dict["code"]}\n</code>\n\n'
        workspace_content += f'<function>\n{state.workspace_dict["function"]}\n</function>\n\n'
        workspace_content += f'<principle>\n{state.workspace_dict["principle"]}\n</principle>\n\n'

    schedule_content = str(state.history_list[-1]['plan'])
    adaptation_content = adaptation

    prompt_text = adapter_prompt.format(
        instruction=instruction_content.strip(),
        analysis=analysis_content.strip(),
        workspace=workspace_content.strip(),
        schedule=schedule_content.strip(),
        adaptation=adaptation_content.strip()
    )
    return prompt_text


def parse_adapter_agent_answer(answer: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    function = safe_extract_from_soup(soup, 'function')
    principle = safe_extract_from_soup(soup, 'principle')
    return code, function, principle


refiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

The core requirements and the expected paradigm are analyzed as follows:

{analysis}

## Reference

According to the requirements, we have retrieved some relevant workflows which may be helpful:

{reference}

## Workspace

The code and annotation of the current workflow you are working on are presented as follows:

{workspace}

## Refinement

An error is detected in the current workflow, which is caused by some bugs in the Python code, such as nested calls and missing parameters. The specific error message is presented as follows:

{refinement}

First, you should explain the reason of the error. Your explanation should be enclosed with "<explanation>" tag. For example: <explanation>The error is caused by the missing input.</explanation>.

After that, you should provide the corrected Python code to formulate the updated workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. You should also avoid reusing the same variable name, even if the variable is temporary. Your code should be enclosed with "<code>" tag. For example: <code>output = node(input)</code>.

Finally, you should provide an annotation as in the reference, including the function and principle of the updated workflow. The function should be enclosed with "<function>" tag. For example: <function>This workflow generates a high-resolution image of a running horse.</function>. The principle should be enclosed with "<principle>" tag. For example: <principle>The workflow first generates a low-resolution image using the text-to-image pipeline and then applies an upscaling module to improve the resolution.</principle>.

Now, provide your explanation, code, and annotation with the required format.
'''


def format_refiner_agent_prompt(state: ComfyAgentState, refinement: str) -> str:
    instruction_content = state.instruction
    analysis_content = state.analysis

    reference_content = ''
    for reference in state.reference_list:
        reference_content += f'- Workflow: {reference.metadata["name"]}\n\n'
        reference_content += f'<code>\n{reference.metadata["code"]}\n</code>\n\n'
        reference_content += f'<function>\n{reference.metadata["function"]}\n</function>\n\n'
        reference_content += f'<principle>\n{reference.metadata["principle"]}\n</principle>\n\n'

    workspace_content = ''
    if len(state.workspace_dict) == 0:
        workspace_content += '- The workspace is empty.'
    else:
        workspace_content += f'<code>\n{state.workspace_dict["code"]}\n</code>\n\n'
        workspace_content += f'<function>\n{state.workspace_dict["function"]}\n</function>\n\n'
        workspace_content += f'<principle>\n{state.workspace_dict["principle"]}\n</principle>\n\n'

    refinement_content = refinement

    prompt_text = refiner_prompt.format(
        instruction=instruction_content,
        analysis=analysis_content,
        reference=reference_content,
        workspace=workspace_content,
        refinement=refinement_content
    )
    return prompt_text


def parse_refiner_agent_answer(answer: str) -> tuple[str, str, str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    explanation = safe_extract_from_soup(soup, 'explanation')
    code = safe_extract_from_soup(soup, 'code')
    function = safe_extract_from_soup(soup, 'function')
    principle = safe_extract_from_soup(soup, 'principle')
    return explanation, code, function, principle


class ComfyPipeline(object):
    def __init__(self, num_references: int = 5, step_limitation: int = 5, debug_limitation: int = 1):
        self.num_references = num_references
        self.step_limitation = step_limitation
        self.debug_limitation = debug_limitation

    def __call__(self, instruction: str):
        # start pipeline
        print(' Task Instruction '.center(80, '-'))
        print(instruction)
        print()

        # analyze instruction
        message = format_analyzer_agent_prompt(instruction)
        print(' Analyzer Prompt '.center(80, '-'))
        print(message)
        print()
        analysis, usage = invoke_completion(message)
        print(' Analyzer Answer '.center(80, '-'))
        print(analysis)
        print(usage)
        print()

        # initialize state
        storage = ReferenceStorage()
        query = f'Instruction: {instruction}\n\nAnalysis: {analysis}'
        reference_list = storage.retrieve(query, count=self.num_references)
        print(' Retrieved Reference '.center(80, '-'))
        for reference in reference_list:
            print(reference.page_content)
            print()
        state = ComfyAgentState(instruction, analysis, reference_list, self.step_limitation)

        # generate workflow
        for step in range(1, self.step_limitation + 1):
            # ----- 1️⃣  Ask the LLM WITH web-search but WITHOUT JSON mode  -----
            message = format_planner_agent_prompt(state)
            raw_answer, usage = invoke_llm_with_websearch(message)

            # ----- 2️⃣  Feed that prose to the adapter for structured output  -----
            planner_schema = (
                'Return a JSON object with keys "thought", "plan", and "action". '
                '"thought": string   # your internal reasoning\n'
                '"plan":   string   # numbered or bulleted steps\n'
                '"action": string   # one of: load … / combine … / adapt … / retrieve … / finish'
            )
            parsed = json_adapter(raw_answer, planner_schema)

            # ----- 3️⃣  Extract & proceed exactly as before  -----
            thought = parsed["thought"]
            plan = parsed["plan"]
            action = parsed["action"]

            command, arguments = parse_planner_agent_action(action)
            state.update_history(thought, plan, action)
            print(' Parsed Thought '.center(80, '-'))
            print(thought)
            print()
            print(' Parsed Plan '.center(80, '-'))
            print(plan)
            print()
            print(' Parsed Action '.center(80, '-'))
            print(action)
            print()

            if command == 'load':
                reference = state.fetch_reference(arguments['name'])
                if reference is None:
                    raise RuntimeError('invalid reference')
                code = reference.metadata['code']
                function = reference.metadata['function']
                principle = reference.metadata['principle']
                state.update_workspace(code, function, principle)

            elif command == 'combine':
                reference = state.fetch_reference(arguments['name'])
                if reference is None:
                    raise RuntimeError('invalid reference')

                message = format_combiner_agent_prompt(state, reference)
                print(' Combiner Prompt '.center(80, '-'))
                print(message)
                print()
                combination, usage = invoke_completion(message)
                print(' Combiner Answer '.center(80, '-'))
                print(combination)
                print(usage)
                print()

                code, function, principle = parse_combiner_agent_answer(combination)
                print(' Parsed Code '.center(80, '-'))
                print(code)
                print()
                print(' Parsed Function '.center(80, '-'))
                print(function)
                print()
                print(' Parsed Principle '.center(80, '-'))
                print(principle)
                print()

                state.update_workspace(code, function, principle)

            elif command == 'adapt':
                message = format_adapter_agent_prompt(state, arguments['prompt'])
                print(' Adapter Prompt '.center(80, '-'))
                print(message)
                print()
                adaptation, usage = invoke_completion(message)
                print(' Adapter Answer '.center(80, '-'))
                print(adaptation)
                print(usage)
                print()

                code, function, principle = parse_adapter_agent_answer(adaptation)
                print(' Parsed Code '.center(80, '-'))
                print(code)
                print()
                print(' Parsed Function '.center(80, '-'))
                print(function)
                print()
                print(' Parsed Principle '.center(80, '-'))
                print(principle)
                print()

                state.update_workspace(code, function, principle)

            elif command == 'retrieve':
                reference_list = storage.retrieve(arguments['prompt'], count=self.num_references)
                print(' Retrieved Reference '.center(80, '-'))
                for reference in reference_list:
                    print(reference.page_content)
                    print()

                state.update_reference(reference_list)

            elif command == 'finish':
                break

            for debug in range(self.debug_limitation + 1):
                try:
                    prompt = parse_code_to_prompt(state.workspace_dict['code'])
                    if prompt is None:
                        raise RuntimeError('failed to parse code')
                    else:
                        break

                except Exception as error:
                    if debug == self.debug_limitation:
                        raise RuntimeError('failed to refine workflow') from error

                    message = format_refiner_agent_prompt(state, str(error))
                    print(' Refiner Prompt '.center(80, '-'))
                    print(message)
                    print()
                    answer, usage = invoke_completion(message)
                    print(' Refiner Answer '.center(80, '-'))
                    print(answer)
                    print(usage)
                    print()

                    explanation, code, function, principle = parse_refiner_agent_answer(answer)
                    print(' Parsed Explanation '.center(80, '-'))
                    print(explanation)
                    print()
                    print(' Parsed Code '.center(80, '-'))
                    print(code)
                    print()
                    print(' Parsed Function '.center(80, '-'))
                    print(function)
                    print()
                    print(' Parsed Principle '.center(80, '-'))
                    print(principle)
                    print()

                    state.update_workspace(code, function, principle)

            if step == self.step_limitation:
                raise RuntimeError('failed to generate workflow')

        # parse code
        prompt = parse_code_to_prompt(state.workspace_dict['code'])
        print(' Parsed Workflow '.center(80, '-'))
        print(json.dumps(prompt, indent=4))
        print()

        return prompt
