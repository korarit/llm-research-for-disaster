import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY
)

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_help_request": {
            "type": "boolean",
            "description": "True if the tweet represents a direct request for rescue, evacuation, medical aid, or immediate basic supplies. False otherwise."
        },
        "category": {
            "type": "string",
            "enum": ["help_request", "other"],
            "description": "Classify as 'help_request' for emergency/relief calls, or 'other' for general updates, weather, wishes, news, and spam."
        }
    },
    "required": ["is_help_request", "category"]
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_disaster",
            "description": "Outputs JSON object fitting the schema for classify_disaster",
            "parameters": CLASSIFICATION_SCHEMA
        }
    }
]
tool_choice = {"type": "function", "function": {"name": "classify_disaster"}}

try:
    response = client.chat.completions.create(
        model="deepseek/deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "You are a disaster emergency dispatcher."},
            {"role": "user", "content": "ช่วยด้วยค่ะ ตอนนี้ติดอยู่ในบ้านน้ำท่วมสูงมาก"}
        ],
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.0
    )
    print("Response object:", response)
    choice = response.choices[0]
    print("Choice object:", choice)
    print("Message object:", choice.message)
    print("Tool calls:", choice.message.tool_calls)
except Exception as e:
    print("Error during call:", e)
