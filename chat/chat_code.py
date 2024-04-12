import openai
import os

# Change the context if you want to change Jarvis' personality
context = "You are Jarvis, Rayan's human assistant. You are a chatbot that answer AI related questions"
conversation = {"Conversation": []}



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt_client = openai.Client(api_key=OPENAI_API_KEY)


class Chat:
    def __init__(self):
        self.context = context
        self.conversation = conversation
        pass
def request_gpt(self, prompt: str) -> str:
    """
    Send a prompt to the GPT-3 API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    response = gpt_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content
