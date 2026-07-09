# แผนการทดลองใช้ LLM ในการคัดแยกข้อความแจ้งเตือนภัยพิบัติ (Experiment 03F) - Few-Shot 2-Agent Sequential Pipeline

เอกสารฉบับนี้กำหนดแผนและแนวทางการทดลองสำหรับ **Experiment 03F** ซึ่งเป็นสถาปัตยกรรมแบบ **เอเจนต์สองขั้นตอนแยกจากกัน (2-Agent / 2-Stage Pipeline)** โดยใช้เทคนิคการเรียนรู้แบบมีตัวอย่างตัวชี้แนะในพรอมต์ (Few-Shot Prompt Learning)

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
- **Agent 1 (Informativeness Filter):** กรองหาความเกี่ยวข้องเกี่ยวกับภัยพิบัติ
- **Agent 2 (Category Classifier):** หาก Agent 1 ตอบว่าเกี่ยวข้อง จึงเรียก Agent 2 มาแยกประเภทหมวดหมู่ย่อย

---

## 4. หลักการออกแบบ Prompt (Rules-First, Examples-as-Anchor)

**แนวทางที่ใช้ใน 03F:**
- **Agent 1** — Gatekeeper system (จาก 03E) + boundary criteria เต็มรูปแบบ + 4 examples (2 clear + 2 edge cases ที่เป็นแต่ละด้าน)
- **Agent 2** — Categorizer system (จาก 03E) + priority-ordered categories พร้อม signal words + 7 examples หลากภัยพิบัติ
- **Anti-anchor disclaimer** — ป้องกัน model ไม่ให้ pattern-match รูปแบบ tweet แทนเข้าใจ taxonomy

---

## 5. รูปแบบคำสั่งแบบมีตัวอย่าง (Few-Shot Prompts)

### 5.1 Agent 1 — Informativeness Filter

**System Prompt:**
```
You are a disaster information gatekeeper. Your ONLY job is to decide whether a tweet contains specific, factual information about a disaster — not to classify its content.

BIAS RULE: When in doubt, lean toward "informative". Only classify as "not_informative" when the tweet is CLEARLY emotional-only, political, or unrelated. It is worse to miss real disaster information than to pass through a borderline case.
```

**User Prompt:**
```
Does this tweet contain SPECIFIC, FACTUAL information about a disaster event?

informative — Mark as informative if the tweet contains ANY of:
  - Specific numbers: casualty counts, displaced persons, missing people, damage estimates
  - Named locations with concrete disaster impact described
  - Factual descriptions of physical destruction (infrastructure, buildings, vehicles, utilities)
  - Reports of rescue, aid, or emergency response operations with details
  - Measurable data: wind speed, magnitude, flood levels, temperature extremes

not_informative — ONLY mark as not_informative if the tweet contains EXCLUSIVELY:
  - Pure emotional expression (prayers, sympathy, fear, grief) with NO factual details
  - Political argument or blame with NO specific disaster impact described
  - Jokes, obvious sarcasm, or clear misinformation
  - Completely off-topic content unrelated to the disaster
  - Vague awareness posts ("Thinking of everyone affected")

REFERENCE EXAMPLES (diverse disaster types — apply criteria above to any tweet):

[Clear not_informative] "Praying for everyone in the typhoon's path. God bless 🙏"
→ not_informative (pure emotional, no factual details)

[Clear informative] "6.2 Earthquake hits Nepal - 150 killed, rescue teams deployed https://t.co/xyz"
→ informative (specific casualty count + response details)

[Edge case → informative] "PHOTOS: Deadly wildfires rage in California https://t.co/td9xT3vXOL"
→ informative (reports deadly wildfire with factual context, despite no exact number)

[Edge case → not_informative] "California wildfire. 4 https://t.co/a8oD5rkDdI"
→ not_informative (no specific factual content, just a fragment + link)

NOTE: Use the criteria above to classify any tweet regardless of similarity to these examples.

Tweet: "{tweet_text}"

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
This tweet has been confirmed as informative about a disaster. Classify it into the SINGLE most dominant humanitarian category.

Apply these rules broadly across all disaster types — not just the examples below:

1. injured_or_dead_people (CHECK FIRST)
   Deaths, fatalities, injuries, casualty counts, hospitalized persons
   Signal words: killed, dead, died, fatalities, casualties, injured, wounded, hospitalized, body count
   ⚠ Takes priority over affected_individuals even if displacement also mentioned

2. missing_or_found_people
   Specific individuals unaccounted for, active search for named/counted persons, confirmed rescues of specific people
   Signal words: missing persons, unaccounted for, search for survivors, found alive, rescued [specific person], no contact with family
   ⚠ About SPECIFIC INDIVIDUALS — not general rescue team deployment

3. affected_individuals
   Displacement and evacuation WITHOUT reported deaths/injuries
   Signal words: displaced, evacuated, evacuees, homeless, stranded, taking shelter, survivors
   ⚠ Only use when NO deaths or injuries are mentioned

4. infrastructure_and_utility_damage
   Damage to built environment structures (not vehicles)
   Signal words: building collapsed, road damaged, bridge failure, power outage, water supply cut, flooded streets, blackout
   ⚠ Do NOT use when vehicles are the primary subject

5. vehicle_damage
   Vehicles as the MAIN focus of the damage reported
   Signal words: car submerged, boats destroyed, vehicles swept away, truck overturned, aircraft grounded
   ⚠ Only when vehicles are the PRIMARY topic, not a side mention

6. rescue_volunteering_or_donation_effort
   Organized emergency response, humanitarian aid, volunteer mobilization
   Signal words: donate, volunteer, aid convoy, rescue team deployed, relief supplies, fundraising, emergency shelter opening
   ⚠ About COLLECTIVE ORGANIZED EFFORTS — not rescue of individual named persons

7. other_relevant_information (USE LAST RESORT ONLY)
   Factual but fits none of the above: weather tracking, earthquake magnitude, storm path, satellite images, warnings without impact
   ⚠ If any specific category above fits, use that instead

REFERENCE EXAMPLES (multiple disaster types):

- injured_or_dead_people → [Wildfire] "Mass Evacuations in California as Wildfires Kill at Least 10"
- injured_or_dead_people → [Earthquake] "6.2 Earthquake hits Nepal - 150 killed, rescue teams deployed"
- missing_or_found_people → [Wildfire] "More than 100 missing persons reports made in California wildfires"
- missing_or_found_people → [Flood] "Family of 5 missing after flash flood swept through their home, search ongoing"
- affected_individuals → [Wildfire] "I just had to evacuate my home in California due to the wildfire."
- affected_individuals → [Hurricane] "Over 3000 evacuees sheltering at Houston Civic Center after Harvey flooding"
- infrastructure_and_utility_damage → [Hurricane] "Power outage affecting 1.2 million homes in Florida after Hurricane Irma"
- infrastructure_and_utility_damage → [Earthquake] "Multiple bridges collapsed in Kathmandu following 7.8 magnitude quake"
- vehicle_damage → [Flood] "Dozens of vehicles swept away as flash flood overtook parking garage in Riyadh"
- vehicle_damage → [Wildfire] "Cars burned with melted rims, trees standing — wildfire path"
- rescue_volunteering_or_donation_effort → [Typhoon] "Red Cross deploying 500 relief workers to typhoon-hit provinces"
- rescue_volunteering_or_donation_effort → [Wildfire] "How to help Napa fire victims: 8 things you can do for Wine Country right now"
- other_relevant_information → [Hurricane] "Hurricane Maria now Category 4 with 130mph winds, expected to hit Puerto Rico Tuesday"
- other_relevant_information → [Earthquake] "USGS: 7.1 magnitude earthquake detected off coast of Japan, tsunami warning issued"

NOTE: Apply the category definitions above to classify any tweet regardless of disaster type or similarity to examples.

Tweet: "{tweet_text}"

Call the function 'classify_category' with your decision.
```

---

## 6. แนวทางการจัดเก็บข้อมูลและประเมินผล
ผลลัพธ์จะถูกจัดเก็บไว้ในโครงสร้างดังนี้:
```text
e:/nlp-for-disaster/exp3F/results/
├── deepseek-v4-flash_temp_results.csv        <- ผลการจำแนกแยกตามอุณหภูมิ
├── typhoon-v2.5_temp_results.csv
├── gemma-4_temp_results.csv
├── model_comparison_metrics.csv              <- เปรียบเทียบ F1-Score ใน Exp 3F
└── confusion_matrices/
```

### 6.1 โครงสร้างของไฟล์ CSV ผลลัพธ์
`tweet_id`, `tweet_text`, `true_text_info`, `true_text_human`, `temperature`, `agent1_predicted_info`, `agent2_predicted_category`, `final_predicted_info`, `final_predicted_category`, `tweet_text_char_count`, `token_in_use`, `token_out_use`, `agent1_latency_seconds`, `agent2_latency_seconds`, `latency_seconds`
