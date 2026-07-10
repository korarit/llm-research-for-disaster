# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 03E-COT) - Improved Zero-Shot 2-Agent Sequential Pipeline with Chain-of-Thought (COT)

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 03E-COT** ซึ่งเป็นสถาปัตยกรรมแบบ **เอเจนต์สองขั้นตอนแยกจากกัน (2-Agent / 2-Stage Pipeline)** ร่วมกับการประยุกต์ใช้ Chain-of-Thought (COT) ภายใต้การปรับปรุงคำสั่งแบบไม่มีตัวอย่าง (Improved Zero-Shot Prompting)

การทดลองนี้ประเมินความแม่นยำ F1-Score และ Latency ของโมเดล MoE ทั้ง 3 รุ่นที่อุณหภูมิการทำงาน (Temperature) 4 ระดับ ได้แก่ **0.0, 0.1, 0.2, 0.3**

---

## 1. โมเดลประมวลผล (LLM Models)
- **deepseek-v4-flash** (OpenRouter: `deepseek/deepseek-v4-flash`)
- **typhoon-v2.5** (OpenTyphoon: `typhoon-v2.5-30b-a3b-instruct`)
- **gemma-4** (OpenRouter: `google/gemma-4-26b-a4b-it`)

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล
- **ที่อยู่ชุดข้อมูล (Dataset Location):** `e:/nlp-for-disaster/dataset/dataset_sample_500.csv`

---

## 3. รูปแบบการทำงานร่วมกันแบบ 2 เอเจนต์ (2-Agent Pipeline)
- **Agent 1 (Informativeness Filter):** กรองแยกแยะทวีตที่เกี่ยวข้องกับภัยพิบัติ โดยทำ CoT ก่อนที่จะตอบพารามิเตอร์ `informativeness`
- **Agent 2 (Category Classifier):** หากเอเจนต์แรกยืนยันว่าเกี่ยวข้อง จะส่งเนื้อหาต่อมาให้เอเจนต์ตัวที่สองทำ CoT เพิ่มเติมวิเคราะห์หาประเภทภัยพิบัติ `category`

---

## 4. รูปแบบคำสั่งและการออกแบบฟังก์ชัน (Prompt & Function Schema Design)

### 4.1 Agent 1 — Informativeness Filter

**ฟังก์ชันเรียกใช้งาน (Filter Function Schema):**
```json
{
    "type": "function",
    "function": {
        "name": "filter_informativeness",
        "description": "Decide whether a tweet contains specific disaster information with reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_reasoning": {
                    "type": "string",
                    "description": "Briefly analyze the raw tweet for disaster indicators (e.g. casualty counts, location impacts, physical damage) and explain your informativeness logic."
                },
                "informativeness": {
                    "type": "string",
                    "enum": ["informative", "not_informative"],
                    "description": "Is the tweet informative about a disaster?"
                }
            },
            "required": ["short_reasoning", "informativeness"]
        }
    }
}
```

**System Prompt:**
```text
You are a disaster information gatekeeper. Your ONLY job is to decide whether a tweet contains specific, factual information about a disaster — not to classify its content.

BIAS RULE: When in doubt, lean toward 'informative'. Only classify as 'not_informative' when the tweet is CLEARLY emotional-only, political, or unrelated. It is worse to miss real disaster information than to pass through a borderline case.
```

**User Prompt:**
```text
Tweet: "{tweet_text}"

Does this tweet contain SPECIFIC, FACTUAL information about a disaster event? Call 'filter_informativeness' with your reasoning and choice.

informative — Mark as informative if the tweet contains ANY of:
  - Specific numbers: casualty counts, displaced persons, missing people, damage estimates
  - Named locations with concrete disaster impact described
  - Factual descriptions of physical destruction (infrastructure, buildings, vehicles, utilities)
  - Reports of rescue, aid, or emergency response operations with details
  - Measurable data: wind speed, magnitude, flood levels, temperature extremes
  - Weather forecasts, warnings, storm tracks, magnitude reports, or direct discussion referencing a specific disaster (e.g., "Prayers for Nepal #earthquake" contains the Nepal earthquake keyword).

not_informative — ONLY mark as not_informative if the tweet contains EXCLUSIVELY:
  - Pure emotional expression (prayers, sympathy, fear, grief) with NO specific disaster references or details (e.g., "Thinking of everyone affected, stay safe").
  - Political argument or blame with NO specific disaster impact described.
  - Jokes, obvious sarcasm, or clear misinformation.
  - Completely off-topic content unrelated to the disaster.

STEPS:
1. Under "short_reasoning", identify critical words and explain why the tweet is informative or not in 1-2 sentences.
2. Call the function 'filter_informativeness' with both your reasoning and informativeness decision.
```

---

### 4.2 Agent 2 — Category Classifier

**ฟังก์ชันเรียกใช้งาน (Classifier Function Schema):**
```json
{
    "type": "function",
    "function": {
        "name": "classify_category",
        "description": "Categorize the disaster tweet into a humanitarian class with reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "short_reasoning": {
                    "type": "string",
                    "description": "Provide a brief 1-2 sentence analysis connecting key tweet clues to the selected humanitarian category."
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
                        "other_relevant_information"
                    ],
                    "description": "The primary dominant humanitarian category representing the tweet."
                }
            },
            "required": ["short_reasoning", "category"]
        }
    }
}
```

**System Prompt:**
```text
You are a disaster content categorizer. The tweet you receive has already been confirmed as containing specific disaster information. Your job is to explain your analysis and identify its PRIMARY humanitarian content category.

CORE PRINCIPLE: Choose the MOST SPECIFIC category that fits the tweet's dominant subject. Use 'other_relevant_information' only as a last resort when no other category applies.
```

**User Prompt:**
```text
Tweet: "{tweet_text}"

This tweet has been confirmed as informative about a disaster. Classify it into the SINGLE most dominant humanitarian category using this priority order:

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

EDGE-CASE RESOLUTION RULES:
- "Evacuees are being given food at the shelter" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "Bridge collapsed, blocking cars" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

STEPS:
1. Under "short_reasoning", explain your reasoning connecting the tweet text clues to one of the categories in 1-2 sentences.
2. Call 'classify_category' with both your reasoning and chosen category.
```

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp3E_COT/results/
├── deepseek-v4-flash_temp_results.csv        <- บันทึกผลลัพธ์และเหตุผลของเอเจนต์แยกตามอุณหภูมิ
├── typhoon-v2.5_temp_results.csv             <- บันทึกผลลัพธ์และเหตุผลของเอเจนต์แยกตามอุณหภูมิ
├── gemma-4_temp_results.csv                  <- บันทึกผลลัพธ์และเหตุผลของเอเจนต์แยกตามอุณหภูมิ
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ระหว่างโมเดล
└── confusion_matrices/                       <- Confusion Matrix ของรอบ COT
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `predicted_info_reasoning`, `predicted_info`, `predicted_cat_reasoning`, `predicted_category`, `mapped_predicted_info`, `mapped_predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `latency_seconds`
