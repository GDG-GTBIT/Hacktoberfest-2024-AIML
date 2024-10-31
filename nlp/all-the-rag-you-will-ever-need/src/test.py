from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()

completion = client.chat.completions.create(
    #model="gpt-3.5-turbo",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Create a masterful haiku about the weather"}
    ]
)

print(completion.choices[0].message)
