# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 01F) - Few-Shot Flat Classification

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 01F** ซึ่งเป็นการคัดแยกประเภทข้อความสถานการณ์ภัยพิบัติด้วยวิธี **เลเยอร์เดียวขั้นตอนเดียว (Single-Layer / Flat Classification)** โดยใช้เทคนิคการเรียนรู้แบบมีตัวอย่างตัวชี้แนะในพรอมต์ (Few-Shot Prompt Learning)

การทดลองนี้ประเมินความแม่นยำ F1-Score และ Latency ของโมเดล MoE ทั้ง 3 รุ่นที่อุณหภูมิการทำงาน (Temperature) 4 ระดับ ได้แก่ **0.0, 0.1, 0.2, 0.3**

---

## 1. โมเดลประมวลผล (LLM Models)
- **deepseek-v4-flash** (OpenRouter: `deepseek/deepseek-v4-flash`)
- **typhoon-v2.5** (OpenTyphoon: `typhoon-v2.5-30b-a3b-instruct`)
- **gemma-4** (OpenRouter: `google/gemma-4-26b-a4b-it`)

---

## 2. แหล่งข้อมูลและการเตรียมข้อมูล
- **ที่อยู่ชุดข้อมูล (Dataset Location):** `e:/nlp-for-disaster/dataset/dataset_sample_500.csv` (ชุดข้อมูลสุ่ม 500 รายการเดียวกันกับการทดลองอื่น ๆ)

---

## 3. สิ่งที่ปรับปรุงจาก Experiment 01 (Zero-Shot เดิม)

| ด้านที่ปรับปรุง | Exp 01 (เดิม) | Exp 01F (ปรับปรุง - Optimized) |
|---|---|---|
| System Prompt | Generic analyst role | บทบาทผู้เชี่ยวชาญด้าน humanitarian disaster triage |
| User Prompt Flow | ไม่มี step-by-step | ยกเลิกโครงสร้าง STEP 1/2 แบบเดิมเพื่อลด Bias และปรับเป็น 1-Step Flat Classification |
| Category Description | อธิบายสั้น 1 บรรทัด | เพิ่ม Signal Words ครบทุก category เพื่อความถูกต้องในการแยกคลาส |
| Reference Examples | ไม่มี | เพิ่มตัวอย่าง Few-shot ที่สะท้อนเกณฑ์การตรวจหาและขอบเขตที่สอดคล้องกับชุดข้อมูลจริง (เช่น ข่าวพยากรณ์อากาศ ความเห็นทางการเมืองที่เอ่ยชื่อพายุ) |

---

## 4. รูปแบบคำสั่งปรับปรุง (Improved Few-Shot Prompt Design)

### 4.1 คำสั่งระบบ (System Instruction)
```
You are a humanitarian disaster information analyst. Your task is to classify social media posts (tweets) collected during disaster events into exactly one category for emergency response analysis.
```

### 4.2 คำสั่งผู้ใช้ (User Prompt Template)
```
Tweet: "{tweet_text}"

Classify the tweet into exactly ONE of the following categories. Choose the category that best represents the primary subject of the tweet:

CATEGORY DEFINITIONS:
1. not_informative: Completely off-topic, spam, irrelevant content, or general personal banter that does not mention or refer to the disaster at all.
2. injured_or_dead_people: Reports of casualties, deaths, fatalities, or injured people. (Signal words: killed, dead, casualties, injured, hospitalized)
3. missing_or_found_people: Reports of individuals or groups who are missing or have been found/rescued. (Signal words: missing, search for, found, rescued)
4. affected_individuals: Evacuees, displaced people, or survivors (without death or injury). (Signal words: evacuated, displaced, homeless, shelter, stranded)
5. infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power lines, or utilities. (Signal words: collapsed, damaged, outage, flooded, blackout)
6. vehicle_damage: Damage to cars, trucks, boats, or planes. (Signal words: car submerged, vehicle damaged)
7. rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, or rescue operations. (Signal words: donate, volunteers, aid, rescue team, relief)
8. other_relevant_information: General news, weather forecasts, warnings, comments, political remarks, or opinions about the disaster that do not fit the specific categories above. (Signal words: warning, forecast, category, magnitude, news, report)

---

EXAMPLES OF CORRECT CLASSIFICATIONS (Pay close attention to boundary cases):

# Example 1: Politically charged comment but explicitly mentions the disaster name
Tweet: "Irma Survivor Tells Trump: Obama Was Playing Golf During The Last Hurricane"
Category: other_relevant_information

# Example 2: General expression of solidarity referencing the disaster
Tweet: "The Prayer Circle: Texans Rebuild After Harvey as a Practice of Faith"
Category: other_relevant_information

# Example 3: Completely off-topic or ambiguous without disaster context
Tweet: "That's cause Steve Harvey did the announcing..."
Category: not_informative

# Example 4: Evacuation/Survival without injuries reported
Tweet: "22K people displaced in Sri Lanka due to being hit by worst flood in decades"
Category: affected_individuals

# Example 5: Casualties and deaths reported
Tweet: "Mass Evacuations in California as Wildfires Kill at Least 10"
Category: injured_or_dead_people

# Example 6: Organized relief effort/Donations
Tweet: "Red Cross is helping people in Houston after Harvey. Donate now!"
Category: rescue_volunteering_or_donation_effort

---

Return your classification by calling the 'classify' function.
```

---

## 5. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp1F/results/
├── deepseek-v4-flash_temp_results.csv        <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── typhoon-v2.5_temp_results.csv             <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── gemma-4_temp_results.csv                  <- บันทึกผลลัพธ์แยกตามอุณหภูมิ (0.0, 0.1, 0.2, 0.3)
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ระหว่างโมเดลและอุณหภูมิใน Exp 1F
└── confusion_matrices/                       <- ภาพ Confusion Matrix แยกรายโมเดลและอุณหภูมิ
```

### 5.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `predicted_category`, `mapped_predicted_info`, `mapped_predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `latency_seconds`
