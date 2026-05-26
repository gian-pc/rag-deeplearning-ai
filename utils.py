import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_with_single_input(prompt, max_tokens=1024, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return {
        "role": "assistant",
        "content": response.choices[0].message.content
    }

def generate_with_multiple_input(messages, max_tokens=1024, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages
    )
    return {
        "role": "assistant",
        "content": response.choices[0].message.content
    }