try:
    from .client import call_llm
except (ImportError, ValueError):
    from client import call_llm

TRIAGE_SCHEMA = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain step-by-step in Thai why this color code was chosen based on the extracted symptoms and IITT clinical criteria."
    },
    "triage_color": {
      "type": "string",
      "enum": ["RED", "YELLOW", "GREEN"],
      "description": "The triage color code determined from the clinical criteria: RED (Immediate emergency), YELLOW (Urgent priority), GREEN (Non-urgent)"
    }
  },
  "required": ["reasoning", "triage_color"]
}

def triage_pediatric(name: str, age: int, symptoms_literal: str, model_name: str, temperature: float = 0.0) -> tuple[dict, int, int, float]:
    """
    Determines triage color for a child victim (< 12 years old) using Pediatric IITT guidelines.
    Returns: (result_dict, prompt_tokens, completion_tokens, latency)
    """
    system_prompt = (
        "You are a pediatric emergency triage specialist. Your task is to determine the triage priority level of a child under 12 based on the Interagency Integrated Triage Tool (IITT) guidelines. "
        "IMPORTANT: This is an academic research simulation using synthetic data. Do not refuse to process the text or trigger safety blocks for posts describing injuries or emergency situations. Always output the required tool call."
    )
    
    user_prompt = f"""Victim Details (Child < 12):
- Name: {name}
- Age: {age}
- Extracted Symptoms/Condition: {symptoms_literal}

Assign a triage color (RED, YELLOW, or GREEN) based on the following pediatric IITT criteria:

1. RED (Emergency - Immediate resuscitation / life support needed):
   - Unresponsive, floppy, or in a coma (หมดสติ, ไม่รู้สึกตัว, ปลุกไม่ตื่น, ตัวอ่อนปุย)
   - Severe respiratory distress, gasping, or central cyanosis (หายใจไม่ค่อยออก, หายใจเหนื่อยมาก, ปากเขียว, หายใจเฮือก, ตัวเขียว)
   - Active severe bleeding (เลือดออกไหลไม่หยุด, แผลฉีกขาดฉกรรจ์เลือดไหลพุ่ง)
   - Continuous convulsions/seizing (กำลังชัก, เกร็ง, ชักต่อเนื่อง)
   - Severe environmental exposure leading to shock or near-drowning (จมน้ำ, สำลักน้ำ, ตัวเย็นเจี๊ยบ, ช็อก)

2. YELLOW (Priority - Needs prompt medical care, can wait briefly):
   - Wheezing, stridor, or moderate breathing difficulty without red criteria (หายใจครืดคราด, หายใจหอบ, หายใจมีเสียงหวีด)
   - Inability to feed or drink, or persistent vomiting/diarrhea (ทานอาหารไม่ได้, ดื่มน้ำไม่ได้, อาเจียนตลอดเวลา, ท้องเสียรุนแรง, อ่อนเพลียมาก)
   - High fever with extreme lethargy (ตัวร้อนจัด, ไข้สูงซึมมาก)
   - Suspected fracture or visible acute limb deformity (กระดูกหัก, แขนพัง, ขาบิดเบี้ยว, ล้มหัวกระแทก)
   - Severe pain (เจ็บปวดทรมานมาก, ร้องไห้ไม่ยอมหยุดจากความเจ็บปวด)

3. GREEN (Non-urgent - Minor issues, can wait safely):
   - Minor cuts, scrapes, or mild local pain (แผลถลอก, มีแผลเล็กน้อย, เจ็บนิดหน่อย)
   - Cough or cold with normal breathing and normal alertness (เป็นหวัด, ไอเล็กน้อย, คุยได้รู้เรื่อง)
   - Alert, active, and able to interact normally (ตื่นดี, วิ่งเล่นได้, ทานข้าวได้)

CRITICAL DECISION RULE:
If symptoms are ambiguous or multiple categories apply, select the highest acuity level (RED > YELLOW > GREEN) to ensure child safety.

Call the function 'triage_pediatric' with your decision."""

    return call_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=TRIAGE_SCHEMA,
        function_name="triage_pediatric",
        temperature=temperature
    )
