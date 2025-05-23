- `GenerateReply`: This node facilitates the generation of replies in a chat context by processing messages sent to and from agents. It leverages conversational agents to simulate interactive dialogues, dynamically generating responses based on the input message and the context of the conversation.
    - Inputs:
        - `recipient` (Required): Specifies the agent intended to receive the message, playing a crucial role in determining the context and content of the generated reply. Type should be `AGENT`.
        - `message` (Required): The message content sent to the recipient, which is used as the basis for generating a reply. Its content directly influences the reply's relevance and coherence. Type should be `STRING`.
        - `sender` (Optional): Identifies the sender of the message, providing additional context for the reply generation process. It's optional and defaults to None if not provided. Type should be `AGENT`.
    - Outputs:
        - `reply`: The generated reply from the recipient agent, based on the input message and the conversation context. Type should be `STRING`.
