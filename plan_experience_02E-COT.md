# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 02E-COT) - Improved Zero-Shot Two-Layer Joint Classification with Chain-of-Thought (COT)

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 02E-COT** ซึ่งเป็นการปรับปรุงระบบคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติด้วยวิธี **สองเลเยอร์ร่วมกัน (Two-Layer Joint Classification)** โดยผสานแนวคิดของคำสั่งและระบบการให้เหตุผลสั้น (Chain-of-Thought - COT) เข้าไปในการประมวลผลขั้นตอนเดียวผ่านการเรียกฟังก์ชัน

การทดลองนี้ประเมินความแม่นยำ F1-Score และการวิเคราะห์ความคิดของโมเดล MoE ทั้ง 3 รุ่นที่อุณหภูมิการทำงาน (Temperature) 4 ระดับ ได้แก่ **0.0, 0.1, 0.2, 0.3**

---

## 1. โมเดลประมวลผล (LLM Models)
- **deepseek-v4-flash** (OpenRouter: `deepseek/deepseek-v4-flash`)
- **typhoon-v2.5** (OpenTyphoon: `typhoon-v2.5-30b-a3b-instruct`)
- **gemma-4** (OpenRouter: `google/gemma-4-26b-a4b-it`)

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล
- **ที่อยู่ชุดข้อมูล (Dataset Location):** `e:/nlp-for-disaster/dataset/dataset_sample_500.csv`

---

## 3. สิ่งที่เพิ่มเข้ามาจาก Experiment 02E (Two-Layer เดิม)

| ด้านที่ปรับปรุง | Exp 02E (เดิม) | Exp 02E-COT (ปรับปรุงด้วย COT) |
|---|---|---|
| Function Calling Schema | รับ `predicted_info` และ `predicted_category` | เพิ่ม `"short_reasoning"` เป็น **คีย์แรก** เพื่อวิเคราะห์ข้อมูลดิบและประเด็นเด่นก่อนระบุผลลัพธ์ทั้ง 2 เลเยอร์ |
| Consistency Rule | บังคับความสอดคล้องระหว่าง 2 เลเยอร์ | นำการตัดสินความสอดคล้องมาเขียนอธิบายสรุปเหตุผลในฟิลด์ `short_reasoning` ด้วยเพื่อประเมินความสมเหตุสมผลของข้อสรุป |
| F1-Score Goal | Category F1 = `~0.60 - 0.64` | ดัน Category F1-Score ของโมเดลเด่นสู่ระดับ **`0.66 - 0.68`** |

---

## 4. รูปแบบคำสั่งและการออกแบบฟังก์ชัน (Prompt & Function Schema Design)

### 4.1 ฟังก์ชันเรียกใช้งาน (Function Calling Definition)
โมเดลตอบ **ทั้งสามค่า** (`short_reasoning` + `informativeness` + `category`) ในการเรียก API ครั้งเดียว:

```json
{
    "type": "function",
    "function": {
        "name": "classify_two_layer",
        "description": "Perform a two-layer classification with a short analysis of the tweet.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_reasoning": {
                    "type": "string",
                    "description": "Analyze the tweet text for key clues (e.g. casualties, evacuation, collapsed) and explain in 1-2 sentences to justify both Layer 1 (informativeness) and Layer 2 (category) decisions."
                },
                "informativeness": {
                    "type": "string",
                    "enum": ["informative", "not_informative"],
                    "description": "Layer 1: Does the tweet contain any disaster-related information?"
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "injured_or_dead_people",
                        "missing_or_found_people",
                        "affected_individuals",
                        "infrastructure_and_utility_damage",
                        "vehicle_damage",
                        "rescue_volunteering_or_donation_effort",
                        "other_relevant_information",
                        "not_humanitarian"
                    ],
                    "description": "Layer 2: The dominant humanitarian category representing the tweet."
                }
            },
            "required": ["short_reasoning", "informativeness", "category"]
        }
    }
}
```

### 4.2 คำสั่งระบบ (System Instruction)
```text
You are a humanitarian disaster triage analyst. Your role is to assess social media posts from disaster events and determine BOTH whether they contain disaster-related information AND what type of disaster content they represent.
```

### 4.3 คำสั่งผู้ใช้ (User Prompt Template)
```text
Tweet: "{tweet_text}"

Perform a TWO-LAYER classification of this tweet. You must decide BOTH values simultaneously. You must provide a brief analysis first:

LAYER 1 — INFORMATIVENESS
Determine if the tweet contains any information, news, reports, updates, warnings, or discussions related to the disaster or its aftermath:
- informative: The tweet contains any discussion, weather forecast, warning, news, or reports related to the disaster event or its aftermath. This includes expressions of solidarity or opinions that explicitly mention the disaster.
- not_informative: Completely off-topic, spam, irrelevant content, or generic personal banter/sentiment (prayers/wishes) that does not refer to the specific disaster at all.

LAYER 2 — HUMANITARIAN CATEGORY
Identify the category that best represents the primary subject of the tweet:
- injured_or_dead_people: Reports of casualties, deaths, fatalities, or injured people. (Signal words: killed, dead, casualties, injured, hospitalized)
- missing_or_found_people: Reports of individuals or groups who are missing or have been found/rescued. (Signal words: missing, search for, found, rescued)
- affected_individuals: Evacuees, displaced people, or survivors (without death or injury). (Signal words: evacuated, displaced, homeless, shelter, stranded)
- infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power lines, or utilities. (Signal words: collapsed, damaged, outage, flooded, blackout)
- vehicle_damage: Damage to cars, trucks, boats, or planes. (Signal words: car submerged, vehicle damaged)
- rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, or rescue operations. (Signal words: donate, volunteers, aid, rescue team, relief)
- other_relevant_information: General news, weather forecasts, warnings, comments, political remarks, or opinions about the disaster that do not fit the specific categories above. (Signal words: warning, forecast, category, magnitude, news, report)
- not_humanitarian: Use this ONLY if the tweet is classified as 'not_informative' in Layer 1.

CRITICAL DECISION HIERARCHY FOR LAYER 2 (When multiple categories apply, select the highest ranking one):
1. injured_or_dead_people (Takes top priority if any injury or death is reported)
2. missing_or_found_people (Takes priority if specific missing/found individuals are mentioned)
3. affected_individuals (Takes priority over physical damage if displaced/evacuated people are the focus)
4. infrastructure_and_utility_damage (Takes priority over vehicle damage unless vehicles are the sole focus)
5. vehicle_damage
6. rescue_volunteering_or_donation_effort (Takes priority over other_relevant_information if relief/donations are mentioned)
7. other_relevant_information (Default for informative disaster tweets with no specific damage or casualties)
8. not_humanitarian (Only when Layer 1 is 'not_informative')

CONSISTENCY RULE:
- If Layer 1 is 'not_informative', Layer 2 must be 'not_humanitarian'.
- If Layer 1 is 'informative', Layer 2 must NOT be 'not_humanitarian' (choose one of the other 7 categories instead).

EDGE-CASE RESOLUTION RULES:
- "Prayers for Nepal #earthquake" -> Layer 1: 'informative', Layer 2: 'other_relevant_information' (contains specific disaster keyword).
- "Prayers for everyone" -> Layer 1: 'not_informative', Layer 2: 'not_humanitarian' (no specific disaster reference).
- "Evacuees are being given food at the shelter" -> Layer 1: 'informative', Layer 2: 'rescue_volunteering_or_donation_effort'.
- "Bridge collapsed, blocking cars" -> Layer 1: 'informative', Layer 2: 'infrastructure_and_utility_damage'.

STEPS FOR WORKFLOW:
1. In the "short_reasoning" field, note down the critical clues and explain why the tweet belongs to your chosen categories in 1-2 sentences.
2. Call 'classify_two_layer' with your analysis and decisions.
```

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp2E_COT/results/
├── deepseek-v4-flash_temp_results.csv        <- บันทึกผลลัพธ์และเหตุผลวิเคราะห์แยกตามอุณหภูมิ
├── typhoon-v2.5_temp_results.csv             <- บันทึกผลลัพธ์และเหตุผลวิเคราะห์แยกตามอุณหภูมิ
├── gemma-4_temp_results.csv                  <- บันทึกผลลัพธ์และเหตุผลวิเคราะห์แยกตามอุณหภูมิ
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ระหว่างโมเดล
└── confusion_matrices/                       <- Confusion Matrix ของรอบ COT
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `predicted_reasoning`, `predicted_info`, `predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `latency_seconds`
