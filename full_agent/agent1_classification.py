try:
    from .client import call_llm
except (ImportError, ValueError):
    from client import call_llm

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

def classify_tweet(text: str, model_name: str, temperature: float = 0.0) -> tuple[dict, int, int, float]:
    """
    Classifies a disaster-related tweet using the Classification Agent.
    Returns: (result_dict, prompt_tokens, completion_tokens, latency)
    """
    system_prompt = (
        "You are a disaster emergency dispatcher. Your task is to classify social media posts (tweets) in Thai to filter out emergency help requests that require urgent rescue or medical dispatch. "
        "IMPORTANT: This is an academic research simulation using synthetic data. Do not refuse to process the text or trigger safety blocks for posts describing injuries or emergency situations. Always output the required tool call."
    )
    
    user_prompt = f"""Tweet: "{text}"

Classify the tweet into exactly ONE category based on the rules below:

CATEGORIES:
1. help_request: Direct calls for rescue, requests to evacuate trapped people, urgent medical aid requests, missing person search requests, or direct requests for emergency food/water.
   - Thai signal words: ช่วยด้วย, ขอความช่วยเหลือ, ติดอยู่, ติดเกาะ, น้ำท่วมสูงมาก, ขออพยพ, ยานอนป่วย, คนแก่ติดอยู่, ท่วมมิดหัว, ต้องการกู้ภัย, ส่งเรือมารับหน่อย, ขอน้ำของประทังชีวิต
2. other: General situation updates, weather reports, warning alerts, official announcements, wishing/praying messages, donation campaigns (raising money/volunteers), or generic spam/news.
   - Thai signal words: อัพเดทน้ำท่วม, พยากรณ์อากาศ, เตือนภัย, ระวังภัย, ประกาศเตือน, ร่วมบริจาคเงินได้ที่, เปิดรับบริจาค, ส่งกำลังใจ, ขอให้ทุกคนปลอดภัย, รายงานสถานการณ์

EDGE-CASE RESOLUTION RULES:
- "ขอรับบริจาคถุงยังชีพไปแจกที่แม่สาย" -> Classify as other. (It's a relief/donation campaign, not a direct emergency victim requesting help).
- "ติดอยู่ในบ้านน้ำท่วมถึงอกพิกัดเกาะทราย ซอย 4 ช่วยด้วยค่ะ" -> Classify as help_request. (Direct victim call).
- "ขอส่งกำลังใจให้เชียงรายรอดพ้นวิกฤตนี้ไปได้โดยเร็ว" -> Classify as other. (Wishes/praying, no emergency dispatch needed).

Call the function 'classify_disaster' with your decision."""

    return call_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=CLASSIFICATION_SCHEMA,
        function_name="classify_disaster",
        temperature=temperature
    )
