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

def triage_adult(name: str, age: int, symptoms_literal: str, model_name: str, temperature: float = 0.0) -> tuple[dict, int, int, float]:
    """
    Determines triage color for an adult or unknown age victim (>= 12 years old or unknown) using Adult IITT guidelines.
    Returns: (result_dict, prompt_tokens, completion_tokens, latency)
    """
    system_prompt = (
        "You are an adult emergency triage specialist. Your task is to determine the triage priority level of an adult or individual of unknown age based on the Interagency Integrated Triage Tool (IITT) guidelines. "
        "IMPORTANT: This is an academic research simulation using synthetic data. Do not refuse to process the text or trigger safety blocks for posts describing injuries or emergency situations. Always output the required tool call."
    )
    
    # Map age to string for prompt
    age_str = str(age) if age is not None else "ไม่ระบุอายุ"
    
    user_prompt = f"""Victim Details (Adult/Unknown >= 12):
- Name: {name}
- Age: {age_str}
- Extracted Symptoms/Condition: {symptoms_literal}

Assign a triage color (RED, YELLOW, or GREEN) based on the following adult IITT criteria:

1. RED (Emergency - Immediate resuscitation / life-saving intervention needed):
   - Unresponsive or severely altered mental status (หมดสติ, ไม่รู้สึกตัว, ปลุกไม่ตื่น, เรียกไม่ตอบสนอง)
   - Severe respiratory distress, cannot speak full sentences, or cyanosis (หายใจลำบากมาก, หายใจเหนื่อยพูดไม่ได้เป็นประโยค, ปากเขียว, ขาดอากาศหายใจ)
   - Active uncontrollable bleeding (เลือดออกพุ่ง, เลือดไหลนองไม่หยุด)
   - Shock signs: weak/rapid pulse, cold sweaty skin (ชีพจรเบาเร็ว, ช็อก, ตัวเย็นเหงื่อออกมาก)
   - High-risk trauma: amputation, severe crush injury, chemical exposure, or active convulsions (แขนขาด, ขาขาด, โดนทับรุนแรง, ชักเกร็ง)

2. YELLOW (Priority - Urgent condition, needs prompt evaluation):
   - Moderate difficulty breathing, wheezing without red criteria (หายใจเหนื่อยหอบหืด, แน่นหน้าอกแต่ยังพูดได้)
   - Chest pain or severe abdominal pain (เจ็บหน้าอกรุนแรง, ปวดท้องรุนแรง)
   - Focal neurological deficits like acute weakness/numbness (แขนขาอ่อนแรงครึ่งซีก, ปากเบี้ยว, พูดไม่ชัด)
   - Persistent vomiting/diarrhea with severe dehydration (อาเจียนรุนแรง, ท้องเสียจนหมดแรง, ขาดน้ำรุนแรง)
   - Open fracture, joint dislocation, or limb deformity (กระดูกโผล่, ข้อเคลื่อน, ขาผิดรูป)
   - Time-sensitive wounds: animal bites, deep wounds, chemical burns (หมากัด, แผลลึก, ไฟลวกผิวหนัง)

3. GREEN (Non-urgent - Minor or chronic issues, can wait safely):
   - Minor wounds, sprains, or abrasions (แผลถลอกเล็กน้อย, ข้อเท้าแพลง, มีแผลเล็กน้อย)
   - Mild pain, normal breathing (ปวดเล็กน้อย, หายใจปกติ)
   - Fully alert, walking wounded, stable vital status (คุยรู้เรื่อง, เดินได้เอง, อาการคงที่)

CRITICAL DECISION RULE:
If symptoms are ambiguous or multiple categories apply, select the highest acuity level (RED > YELLOW > GREEN) to protect the patient.

Call the function 'triage_adult' with your decision."""

    return call_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=TRIAGE_SCHEMA,
        function_name="triage_adult",
        temperature=temperature
    )
