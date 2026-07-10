# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 01E-COT) - Improved Zero-Shot Flat Classification with Chain-of-Thought (COT)

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 01E-COT** ซึ่งเป็นการคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติด้วยวิธี **เลเยอร์เดียวขั้นตอนเดียว (Single-Layer / Flat Classification)** โดยใช้การปรับปรุงคำสั่งร่วมกับการบังคับประมวลผลความคิดและเหตุผลในตัว (Chain-of-Thought - COT) ผ่านโครงสร้างพารามิเตอร์ของ Function Calling

การทดลองนี้ประเมินความแม่นยำ F1-Score และปริมาณโทเค็นที่ใช้งานของโมเดล MoE ทั้ง 3 รุ่นที่อุณหภูมิการทำงาน (Temperature) 4 ระดับ ได้แก่ **0.0, 0.1, 0.2, 0.3**

---

## 1. โมเดลประมวลผล (LLM Models)
- **deepseek-v4-flash** (OpenRouter: `deepseek/deepseek-v4-flash`)
- **typhoon-v2.5** (OpenTyphoon: `typhoon-v2.5-30b-a3b-instruct`)
- **gemma-4** (OpenRouter: `google/gemma-4-26b-a4b-it`)

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล
- **ที่อยู่ชุดข้อมูล (Dataset Location):** `e:/nlp-for-disaster/dataset/dataset_sample_500.csv` (ชุดข้อมูลสุ่ม 500 รายการเดียวกันกับการทดลองอื่น ๆ)

---

## 3. สิ่งที่เพิ่มเข้ามาจาก Experiment 01E (Zero-Shot ดั้งเดิม)

| ด้านที่ปรับปรุง | Exp 01E (เดิม) | Exp 01E-COT (ทำลายจุดบอดด้วย COT) |
|---|---|---|
| Tool/Function Schema | รับพารามิเตอร์เฉพาะผลลัพธ์สุดท้าย `category` | เพิ่มพารามิเตอร์ `"short_reasoning"` เป็น **คีย์แรก** เพื่อบังคับให้โมเดลประมวลผลเหตุผลเชิงตรรกะก่อนเลือกคำตอบ |
| Prompt Guidance | สั่งให้วิเคราะห์แบบ Direct Triage | บังคับลำดับความคิด (Sequential Reasoning) ให้อิงตามเบาะแส คำสำคัญ หรือตัวบ่งชี้ความสูญเสียก่อนทำการเคาะหมวดหมู่ |
| F1-Score Goal | Category F1 = `~0.61 - 0.62` | ดัน Category F1-Score สู่เกณฑ์เป้าหมาย **`0.64 - 0.66`** |

---

## 4. รูปแบบคำสั่งและการออกแบบฟังก์ชัน (Prompt & Function Schema Design)

### 4.1 ฟังก์ชันเรียกใช้งาน (Function Calling Definition)
โมเดลจะต้องถูกจำกัดให้ทำการประมวลผลผ่านพารามิเตอร์ `short_reasoning` เป็นคีย์แรก และตามด้วย `category` ดังนี้:

```json
{
    "type": "function",
    "function": {
        "name": "classify",
        "description": "Classify the disaster-related tweet and provide a brief reasoning analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_reasoning": {
                    "type": "string",
                    "description": "Briefly analyze the tweet text to identify critical clues (e.g. casualties, evacuation, collapsed) and explain in 1-2 sentences why it belongs to the chosen category."
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "not_informative",
                        "affected_individuals",
                        "infrastructure_and_utility_damage",
                        "injured_or_dead_people",
                        "missing_or_found_people",
                        "other_relevant_information",
                        "rescue_volunteering_or_donation_effort",
                        "vehicle_damage"
                    ],
                    "description": "The primary dominant humanitarian category representing the tweet."
                }
            },
            "required": ["short_reasoning", "category"]
        }
    }
}
```

### 4.2 คำสั่งระบบ (System Instruction)
```text
You are a humanitarian disaster information analyst. Your task is to analyze social media posts (tweets) collected during disasters, explain your reasoning, and classify them into exactly one category for emergency response.
```

### 4.3 คำสั่งผู้ใช้ (User Prompt Template)
```text
Tweet: "{tweet_text}"

Classify this tweet into the SINGLE most dominant humanitarian category using this priority order:

1. injured_or_dead_people (CHECK FIRST)
   Deaths, fatalities, injuries, casualty counts, hospitalized persons
   Signal words: killed, dead, died, fatalities, casualties, injured, wounded, hospitalized, body count
   ⚠ Takes priority over affected_individuals and infrastructure damage if any death or injury is mentioned.

2. missing_or_found_people
   Specific individuals unaccounted for, active search for named/counted persons, confirmed rescues of specific people
   Signal words: missing persons, unaccounted for, search for survivors, found alive, rescued [specific person], no contact with family
   ⚠ About SPECIFIC INDIVIDUALS — not general rescue team deployment.

3. affected_individuals
   Displacement and evacuation WITHOUT reported deaths/injuries
   Signal words: displaced, evacuated, evacuees, homeless, stranded, taking shelter, survivors
   ⚠ Only use when NO deaths or injuries are mentioned.

4. infrastructure_and_utility_damage
   Damage to built environment structures (not vehicles)
   Signal words: building collapsed, road damaged, bridge failure, power outage, water supply cut, flooded streets, blackout, structural damage
   ⚠ Do NOT use when vehicles are the primary subject.

5. vehicle_damage
   Vehicles as the MAIN focus of the damage reported
   Signal words: car submerged, boats destroyed, vehicles swept away, truck overturned, aircraft grounded
   ⚠ Only when vehicles are the PRIMARY topic, not a side mention.

6. rescue_volunteering_or_donation_effort
   Organized emergency response, humanitarian aid, volunteer mobilization, emergency helpline sharing, relief distribution
   Signal words: donate, volunteers, aid, rescue team, relief, relief goods, aid distribution
   ⚠ About COLLECTIVE ORGANIZED EFFORTS — not rescue of individual named persons.

7. other_relevant_information (USE LAST RESORT ONLY)
   Factual but fits none of the above: weather tracking, earthquake magnitude, storm path, satellite images, official warnings without impact details, general news/opinions/expressions of solidarity that mention a specific disaster.
   ⚠ If any specific category above fits, use that instead.

8. not_informative
   Completely off-topic, spam, advertisements, jokes, or generic emotional posts (prayers, wishes, thoughts) that do NOT mention any specific disaster or aftermath details.

EDGE-CASE RESOLUTION RULES:
- "Evacuees are being given food at the shelter" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "Bridge collapsed, blocking cars" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

STEPS:
1. Under "short_reasoning", identify critical words (e.g. death counts, damaged roads) and explain why the tweet belongs to your chosen category in 1-2 sentences.
2. Call the function 'classify' with both your reasoning and chosen category.
```

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp1E_COT/results/
├── deepseek-v4-flash_temp_results.csv        <- บันทึกผลลัพธ์และเหตุผลแยกตามอุณหภูมิ
├── typhoon-v2.5_temp_results.csv             <- บันทึกผลลัพธ์และเหตุผลแยกตามอุณหภูมิ
├── gemma-4_temp_results.csv                  <- บันทึกผลลัพธ์และเหตุผลแยกตามอุณหภูมิ
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ระหว่างโมเดลในรอบ COT
└── confusion_matrices/                       <- Confusion Matrix ของรอบ COT
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `predicted_reasoning`, `predicted_category`, `mapped_predicted_info`, `mapped_predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `latency_seconds`
