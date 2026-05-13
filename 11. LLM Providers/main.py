from anthropic import Anthropic
import os
from dotenv import load_dotenv

SYSTEM_MESSAGE = "You are a chatbot. You will have a conversation with a user. Be friendly and concise"

if __name__ == "__main__":
    load_dotenv()
    URL = os.environ.get('API_BASE_URL')
    KEY = os.environ.get('API_KEY')
    MODEL = os.environ.get('MODEL')

    client = Anthropic(
        base_url=URL,
        api_key=KEY,
    )

    print(f"Chatting with {MODEL} model at {URL}\n")

    messages = []
    while True:
        message = input("> ")
        messages.append({'role': 'user', 'content': message})
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_MESSAGE,
            messages=messages,
        )
        reply = response.content[0].text
        messages.append({'role': 'assistant', 'content': reply})
        print(reply)