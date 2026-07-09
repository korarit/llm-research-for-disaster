# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 02F) - Few-Shot Two-Layer Joint Classification

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 02F** ซึ่งเป็นการปรับปรุงระบบคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติด้วยวิธี **สองเลเยอร์ร่วมกัน (Two-Layer Joint Classification)** โดยใช้เทคนิคการเรียนรู้แบบมีตัวอย่างตัวชี้แนะในพรอมต์ (Few-Shot Prompt Learning)

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

## 3. สิ่งที่ปรับปรุงจาก Experiment 02 (Zero-Shot)

| ด้านที่ปรับปรุง | Exp 02 | Exp 02F (ปรับปรุง - Optimized) |
|---|---|---|
| System Prompt | Generic triage role | บทบาทผู้เชี่ยวชาญ triage ที่ลบ Bias อคติที่มุ่งเน้นการคัดทิ้งข้อมูลออก |
| Layer 1 Informativeness | เน้นเฉพาะเจาะจง (actionable) | ปรับคำจำกัดความขยายให้รวมข้อมูล ข่าวสาร บทวิเคราะห์ ความเห็นพายุเพื่อลดคลาสยึดติด |
| Layer 2 Humanitarian | ลำดับการเลือกมีลำดับความสำคัญ | ใช้ระบบสองเลเยอร์แบบไร้ Bias ควบคู่กับ Consistency Rule เพื่อป้องกันข้อมูลขัดแย้ง |
| Reference Examples | ไม่มี | เพิ่ม Few-shot 6 ข้อแบบเดียวกับ 1F โดยจัดรูปแบบเอาต์พุตให้ตรงกับ Schema สองเลเยอร์ |

---

## 4. รูปแบบคำสั่งปรับปรุง (Improved Few-Shot Prompt Design)

### 4.1 คำสั่งระบบ (System Instruction)
```
You are a humanitarian disaster triage analyst. Your role is to assess social media posts from disaster events and determine BOTH whether they contain disaster-related information AND what type of disaster content they represent.
```

### 4.2 คำสั่งผู้ใช้ (User Prompt Template)
```
Tweet: "{tweet_text}"

Perform a TWO-LAYER classification of this tweet. You must decide BOTH values simultaneously:

LAYER 1 — INFORMATIVENESS
Determine if the tweet contains any information, news, reports, updates, warnings, or discussions related to the disaster or its aftermath:
- informative: The tweet contains any discussion, weather forecast, warning, news, or reports related to the disaster event or its aftermath. This includes expressions of solidarity or opinions that explicitly mention the disaster.
- not_informative: Completely off-topic, spam, irrelevant content, or personal banter that does not mention or refer to the disaster at all.

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

CONSISTENCY RULE:
- If Layer 1 is 'not_informative', Layer 2 must be 'not_humanitarian'.
- If Layer 1 is 'informative', Layer 2 must NOT be 'not_humanitarian' (choose one of the other 7 categories instead).

---

EXAMPLES OF CORRECT CLASSIFICATIONS (Pay close attention to boundary cases):

# Example 1: Politically charged comment but explicitly mentions the disaster name
Tweet: "Irma Survivor Tells Trump: Obama Was Playing Golf During The Last Hurricane"
Informativeness: informative
Category: other_relevant_information

# Example 2: General expression of solidarity referencing the disaster
Tweet: "The Prayer Circle: Texans Rebuild After Harvey as a Practice of Faith"
Informativeness: informative
Category: other_relevant_information

# Example 3: Completely off-topic or ambiguous without disaster context
Tweet: "That's cause Steve Harvey did the announcing..."
Informativeness: not_informative
Category: not_humanitarian

# Example 4: Evacuation/Survival without injuries reported
Tweet: "22K people displaced in Sri Lanka due to being hit by worst flood in decades"
Informativeness: informative
Category: affected_individuals

# Example 5: Casualties and deaths reported
Tweet: "Mass Evacuations in California as Wildfires Kill at Least 10"
Informativeness: informative
Category: injured_or_dead_people

# Example 6: Organized relief effort/Donations
Tweet: "Red Cross is helping people in Houston after Harvey. Donate now!"
Informativeness: informative
Category: rescue_volunteering_or_donation_effort

---

Return classification by calling the function 'classify_two_layer' with both values.
```

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp2F/results/
├── deepseek-v4-flash_temp_results.csv        <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── typhoon-v2.5_temp_results.csv             <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── gemma-4_temp_results.csv                  <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ระหว่างโมเดลและอุณหภูมิใน Exp 2F
└── confusion_matrices/                       <- ภาพ Confusion Matrix แยกรายโมเดลและอุณหภูมิ
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `predicted_info`, `predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `latency_seconds`
