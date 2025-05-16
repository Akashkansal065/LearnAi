import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
