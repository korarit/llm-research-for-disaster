# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 03E) - Improved Zero-Shot 2-Agent Sequential Pipeline

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 03E** ซึ่งเป็นสถาปัตยกรรมแบบ **เอเจนต์สองขั้นตอนแยกจากกัน (2-Agent / 2-Stage Pipeline)** โดยใช้การปรับปรุงคำสั่งแบบไม่มีตัวอย่าง (Improved Zero-Shot Prompting)

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

## 3. รูปแบบการทำคลาสสิฟิเคชันแบบ 2 ขั้นตอน (2-Stage Cascade)
- **Agent 1 (Informativeness Filter):** กรองหาความเกี่ยวข้องเกี่ยวกับภัยพิบัติ (`informative` / `not_informative`)
- **Agent 2 (Category Classifier):** หาก Agent 1 ตอบว่าเกี่ยวข้อง จึงเรียก Agent 2 มาแยกประเภทหมวดหมู่ย่อย

---

## 4. สิ่งที่ปรับปรุงจาก Experiment 03 (Zero-Shot เดิม)

| ด้านที่ปรับปรุง | Exp 03 (เดิม) | Exp 03E (ปรับปรุง) |
|---|---|---|
| Agent 1 System Prompt | Generic analyst role | ระบุบทบาท "gatekeeper" ชัดเจน + เน้น err on informative ถ้าไม่แน่ใจ |
| Agent 1 User Prompt | อธิบายสั้น 2 บรรทัด | เพิ่ม Positive/Negative examples แบบ bullet + boundary cases |
| Agent 2 System Prompt | Generic analyst role | ระบุว่าข้อความที่รับมาผ่าน filter แล้ว — ให้ focus เฉพาะหมวดหมู่ |
| Agent 2 User Prompt | ลิสต์สั้น ไม่มี signal words | เพิ่ม Signal Words + ⚠ Disambiguation ครบทุก category + priority order |

---

## 5. รูปแบบคำสั่งแบบปรับปรุง (Improved Zero-Shot Prompts)

### 5.1 Agent 1 — Informativeness Filter

**System Prompt:**
```
You are a disaster information gatekeeper. Your ONLY job is to decide whether a tweet contains specific, factual information about a disaster — not to classify its content.

BIAS RULE: When in doubt, lean toward "informative". Only classify as "not_informative" when the tweet is CLEARLY emotional-only, political, or unrelated. It is worse to miss real disaster information than to pass through a borderline case.
```

**User Prompt:**
```
Tweet: "{tweet_text}"

Does this tweet contain SPECIFIC, FACTUAL information about a disaster event?

▶ informative — Mark as informative if the tweet contains ANY of:
  • Specific numbers: casualty counts, displaced persons, missing people, damage estimates
  • Named locations with concrete disaster impact described
  • Factual descriptions of physical destruction (infrastructure, buildings, vehicles, utilities)
  • Reports of rescue, aid, or emergency response operations with details
  • Measurable data: wind speed, magnitude, flood levels, temperature extremes

▶ not_informative — ONLY mark as not_informative if the tweet contains EXCLUSIVELY:
  • Pure emotional expression (prayers, sympathy, fear, grief) with NO factual details
  • Political argument or blame with NO specific disaster impact described
  • Jokes, obvious sarcasm, or clear misinformation
  • Completely off-topic content unrelated to the disaster
  • Vague awareness posts ("Thinking of everyone affected")

Call the function 'filter_informativeness' with your decision.
```

---

### 5.2 Agent 2 — Category Classifier

**System Prompt:**
```
You are a disaster content categorizer. The tweet you receive has already been confirmed as containing specific disaster information. Your job is to identify its PRIMARY humanitarian content category.

CORE PRINCIPLE: Choose the MOST SPECIFIC category that fits the tweet's dominant subject. Use "other_relevant_information" only as a last resort when no other category applies.
```

**User Prompt:**
```
Tweet: "{tweet_text}"

This tweet has been confirmed as informative about a disaster. Classify it into the SINGLE most dominant humanitarian category using this priority order:

① injured_or_dead_people  ← CHECK FIRST
   Deaths, fatalities, injuries, casualty counts, hospitalized persons
   Signal words: killed, dead, died, fatalities, casualties, injured, wounded, hospitalized, body count
   ⚠ Takes priority over affected_individuals even if displacement also mentioned

② missing_or_found_people
   Specific individuals unaccounted for, active search for named/counted persons, confirmed rescues of specific people
   Signal words: missing persons, unaccounted for, search for survivors, found alive, rescued [specific person], no contact with family
   ⚠ About SPECIFIC INDIVIDUALS — not general rescue team deployment

③ affected_individuals
   Displacement and evacuation WITHOUT reported deaths/injuries
   Signal words: displaced, evacuated, evacuees, homeless, stranded, taking shelter, survivors
   ⚠ Only use when NO deaths or injuries are mentioned

④ infrastructure_and_utility_damage
   Damage to built environment structures (not vehicles)
   Signal words: building collapsed, road damaged, bridge failure, power outage, water supply cut, flooded streets, blackout, structural damage
   ⚠ Do NOT use when vehicles are the primary subject

⑤ vehicle_damage
   Vehicles as the MAIN focus of the damage reported
   Signal words: car submerged, boats destroyed, vehicles swept away, truck overturned, aircraft grounded
   ⚠ Only when vehicles are the PRIMARY topic, not a side mention

⑥ rescue_volunteering_or_donation_effort
   Organized emergency response, humanitarian aid, volunteer mobilization
   Signal words: donate, volunteer, aid convoy, rescue team deployed, relief supplies, fundraising, emergency shelter opening
   ⚠ About COLLECTIVE ORGANIZED EFFORTS — not rescue of individual named persons

⑦ other_relevant_information  ← USE LAST RESORT ONLY
   Factual but fits none of the above: weather tracking, earthquake magnitude, storm path, satellite images, official warnings without impact details
   ⚠ If any specific category above fits, use that instead

Call the function 'classify_category' with your decision.
```

---

## 6. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp3E/results/
├── deepseek-v4-flash_temp_results.csv        <- ผลการจำแนกแยกตามอุณหภูมิ
├── typhoon-v2.5_temp_results.csv
├── gemma-4_temp_results.csv
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ใน Exp 3E
└── confusion_matrices/
```

### 6.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `agent1_predicted_info`, `agent2_predicted_category`, `final_predicted_info`, `final_predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `agent1_latency_seconds`, `agent2_latency_seconds`, `latency_seconds`
