import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")

client = OpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=TYPHOON_API_KEY
)

NER_SCHEMA = {
  "type": "object",
  "properties": {
    "message_more_detail": {
      "type": "string",
      "description": "Brief summary of the disaster incident details in Thai"
    },
    "contact_victim": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": ["string", "null"], "description": "Full name or first name if found, otherwise null" },
          "nickname": { "type": ["string", "null"], "description": "Nickname if found, otherwise null" },
          "phone": { "type": ["string", "null"], "description": "Phone number found in the tweet, otherwise null" }
        },
        "required": ["name", "nickname", "phone"]
      }
    },
    "contact_reporter": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": { "type": ["string", "null"], "description": "Full name or first name if found, otherwise null" },
          "nickname": { "type": ["string", "null"], "description": "Nickname if found, otherwise null" },
          "phone": { "type": ["string", "null"], "description": "Phone number found in the tweet, otherwise null" }
        },
        "required": ["name", "nickname", "phone"]
      }
    },
    "victims": {
      "type": "object",
      "properties": {
        "dead": { "type": "integer", "description": "Number of dead people explicitly reported" },
        "critical": { "type": "integer", "description": "Number of people trapped, missing, in severe danger or severely injured" },
        "urgent": { "type": "integer", "description": "Number of injured or sick people needing prompt assistance" },
        "safe": { "type": "integer", "description": "Number of people reported safe/evacuated" },
        "child": { "type": "integer", "description": "Number of children affected (including infants)" },
        "bedridden": { "type": "integer", "description": "Number of bedridden patients affected" }
      },
      "required": ["dead", "critical", "urgent", "safe", "child", "bedridden"]
    },
    "items": {
      "type": "object",
      "properties": {
        "firstAid": { "type": "integer", "description": "Quantity/Need of first-aid kits or medicine (1 if needed but quantity not specified)" },
        "food": { "type": "integer", "description": "Quantity/Need of food/drinking water (1 if needed but quantity not specified)" },
        "energy": { "type": "integer", "description": "Quantity/Need of flashlights, powerbanks, candles, or backup power (1 if needed but quantity not specified)" }
      },
      "required": ["firstAid", "food", "energy"]
    },
    "coordinates": {
      "type": "object",
      "properties": {
        "name": { "type": ["string", "null"], "description": "Specific location name, landmark, road, or sub-district name mentioned in the tweet" },
        "google_map_url": { "type": ["string", "null"], "description": "Google Maps URL if found, otherwise null" },
        "lat": { "type": "number", "description": "Latitude coordinate if found, otherwise 0.0" },
        "lng": { "type": "number", "description": "Longitude coordinate if found, otherwise 0.0" }
      },
      "required": ["name", "google_map_url", "lat", "lng"]
    }
  },
  "required": [
    "message_more_detail",
    "contact_victim",
    "contact_reporter",
    "victims",
    "items",
    "coordinates"
  ]
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_information",
            "description": "Outputs JSON object fitting the schema for extract_information",
            "parameters": NER_SCHEMA
        }
    }
]
tool_choice = {"type": "function", "function": {"name": "extract_information"}}

tweet = """ช่วยด้วยค่ะ! ตอนนี้เดือดร้อนมาก 🚨 พิกัด 15.0407, 99.7118 มีคนป่วยหลายคนเลยค่ะ 
ตอนนี้มีพี่ผู้ชายมีอาการปวดท้องรุนแรงมาก ทนไม่ไหว (คนไข้ทั่วไปอายุน้อยกว่า 50 ปี) อีกคนเป็นหญิงอายุ 48 ปี มีอาการปวดหัวข้างเดียวรุนแรงมากเหมือนหัวจะระเบิด คลื่นไส้เจ็บกระบอกตา แล้วก็น้องอานวา อายุ 11 ขวบ น้องบ่นเจ็บคอเล็กน้อย ไม่มีไข้อ่อนๆ กินน้ำและอาหารได้ปกติ 

ตอนนี้ไฟดับมืดไปหมดเลยค่ะ ขอไฟฉายหรืออุปกรณ์ให้แสงสว่างหน่อยนะคะ 🔦 ติดต่อคุณสุพิชชา จันทร์ดวงศรี ได้ที่เบอร์ 0670889523 หรือติดต่อหยาดน้ำได้เลยค่ะ 😭🙏"""

try:
    response = client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct",
        messages=[
            {"role": "system", "content": "You are an expert disaster response information analyst."},
            {"role": "user", "content": tweet}
        ],
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.0
    )
    print("Response object:", response)
    choice = response.choices[0]
    print("Message content:", choice.message.content)
    print("Tool calls:", choice.message.tool_calls)
except Exception as e:
    print("Error:", e)
