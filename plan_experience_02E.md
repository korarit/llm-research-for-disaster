# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 02E) - Improved Zero-Shot Two-Layer Joint Classification

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 02E** ซึ่งเป็นการปรับปรุงระบบคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติด้วยวิธี **สองเลเยอร์ร่วมกัน (Two-Layer Joint Classification)** โดยใช้การปรับปรุงคำสั่งแบบไม่มีตัวอย่าง (Improved Zero-Shot Prompting)

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

## 3. สิ่งที่ปรับปรุงจาก Experiment 02 (Zero-Shot เดิม)

| ด้านที่ปรับปรุง | Exp 02 (เดิม) | Exp 02E (ปรับปรุง - Optimized) |
|---|---|---|
| System Prompt | Generic analyst role | บทบาทผู้เชี่ยวชาญด้าน humanitarian triage |
| Informativeness Decision | อธิบายสั้น ๆ | ขยายนิยามของ `informative` ให้ครอบคลุมข่าวสาร คำเตือน และคำอภิปรายเกี่ยวกับภัยพิบัติทั้งหมดเพื่อไม่ให้ตกหล่น (Recall สูงขึ้น) |
| Category Description | ลิสต์สั้น ไม่มี signal words | เพิ่ม Signal Words ครบทุก category เพื่อช่วยในการเลือกคลาส |
| Consistency Rule | ไม่มี | บังคับความสอดคล้อง (Consistency Rule) ระหว่าง Layer 1 และ Layer 2 เพื่อความถูกต้องของผลลัพธ์ |

---

## 4. รูปแบบคำสั่งปรับปรุง (Improved Zero-Shot Prompt Design) - promptV3

โมเดลตอบ **ทั้งสองค่า** (`informativeness` + `category`) ในการเรียก API ครั้งเดียว ผ่าน Function Calling

### 4.1 คำสั่งระบบ (System Instruction)
```text
You are a humanitarian disaster triage analyst. Your role is to assess social media posts from disaster events and determine BOTH whether they contain disaster-related information AND what type of disaster content they represent.
```

### 4.2 คำสั่งผู้ใช้ (User Prompt Template)
```text
Tweet: "{tweet_text}"

Perform a TWO-LAYER classification of this tweet. You must decide BOTH values simultaneously:

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

Return classification by calling the function 'classify_two_layer' with both values.
```

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp2E/results/
├── deepseek-v4-flash_temp_results.csv        <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── typhoon-v2.5_temp_results.csv             <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── gemma-4_temp_results.csv                  <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ระหว่างโมเดลและอุณหภูมิใน Exp 2E
└── confusion_matrices/                       <- ภาพ Confusion Matrix แยกรายโมเดลและอุณหภูมิ
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `predicted_info`, `predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `latency_seconds`
