# แผนการแปลชุดข้อมูลภัยพิบัติสู่ภาษาไทย (CrisisMMD Translation to Thai - Plan)

เอกสารฉบับนี้กำหนดแผนและแนวทางการจัดเตรียมชุดข้อมูลภัยพิบัติแปลไทย **plan_CrisisMMD_to_thai.md** โดยทำการสุ่มตัวอย่างข้อความทวีตใหม่จากชุดข้อมูลอ้างอิงสากล `CrisisMMD_v2.0` นำมาแปลภาษาให้มีความเป็นธรรมชาติ เข้ากับบริบทคำพูดของคนไทย แต่ยังคงความหมายและโครงสร้างดั้งเดิมอย่างครบถ้วนสำหรับใช้ในภารกิจจัดหมวดหมู่ภัยพิบัติ

---

## 1. วัตถุประสงค์ (Objectives)
- จัดทำชุดข้อมูลข้อความภัยพิบัติภาษาไทยมาตรฐานจำนวน **1,000 รายการ** ที่แปลมาจากต้นฉบับภาษาอังกฤษของ CrisisMMD_v2.0
- ใช้โมเดล **`google/gemini-3.1-flash-lite`** (หรือรุ่นที่เหมาะสมในการแปลความเร็วสูง) เพื่อทำการแปลงภาษาแบบรักษาบริบท
- ออกแบบคำสั่ง Prompt ในการแปลโดยเน้นให้โมเดลรังสรรค์สำนวนการเขียนเหมือนคนไทยเขียนเองโดยธรรมชาติ บนแพลตฟอร์มโซเชียลมีเดีย แต่ต้องไม่สูญเสียความหมายหรือสถิติข้อมูลที่เป็นเกณฑ์การจัดหมวดหมู่ภัยพิบัติ (Informativeness & Category Context)

---

## 2. ขั้นตอนการกรองและการสุ่มข้อมูลตัวอย่าง (Filtering & Fallback Sampling Strategy)
เพื่อรักษาคุณภาพข้อมูล บริบทของหมวดหมู่ และความสมดุลของข้อมูลชนกลุ่มน้อย (Minority Classes) ให้สอดคล้องกับการรักษาสัดส่วนเดิมของชุดข้อมูล 500 รายการ (แต่ขยายสเกลเป็น 1,000 รายการแบบเท่าตัว) สคริปต์สุ่มตัวอย่างข้อมูลจะดำเนินงานด้วยระบบ **สุ่มยืมเป็นระดับขั้น (Tiered Fallback Sampling)** ดังนี้:

### 2.1 กำหนดเป้าหมายรายหมวดหมู่ (Target Size per Category)
กำหนดเป้าหมายการกระจายตัวของคลาสในชุดข้อมูล 1,000 รายการ โดยขยายเป็น 2 เท่าจากชุดข้อมูล 500 รายการเดิม เพื่อให้วัดผลประสิทธิภาพของแต่ละหมวดหมู่ย่อยได้อย่างเสถียร:
- `other_relevant_information`: 304 รายการ
- `not_humanitarian`: 250 รายการ
- `rescue_volunteering_or_donation_effort`: 190 รายการ
- `infrastructure_and_utility_damage`: 90 รายการ
- `affected_individuals`: 52 รายการ
- `injured_or_dead_people`: 52 รายการ
- `vehicle_damage`: 32 รายการ
- `missing_or_found_people`: 30 รายการ

### 2.2 ลำดับขั้นการผ่อนปรนเกณฑ์คัดเลือกข้อมูล (Tiered Fallback Tiers)
สำหรับแต่ละหมวดหมู่ ระบบจะพยายามค้นหาผู้สมัคร (Candidates) จากเกณฑ์ที่เข้มงวดที่สุดก่อน หากไม่ครบจำนวนเป้าหมายของหมวดหมู่นั้นๆ ระบบจะขยับไปดึงตัวเลือกเพิ่มจากเกณฑ์ขั้นถัดไปตามลำดับชั้น:
- **Tier 1 (เกณฑ์เข้มข้นสูงสุด):** เลือกเฉพาะข้อมูลทวีตที่มีการป้ายกำกับข้อความและภาพตรงกัน (`text_human == image_human`) AND ความยาวข้อความตั้งแต่ **70 ตัวอักษรขึ้นไป** AND **ไม่ซ้ำซ้อน** กับชุดข้อมูล 500 เดิม (`~isin(prev_500_ids)`)
- **Tier 2 (ลดเกณฑ์ความยาว):** บังคับ Label ตรงกันและไม่ซ้ำชุดเดิม แต่ลดเกณฑ์ความยาวเป็น **50 ตัวอักษรขึ้นไป**
- **Tier 3 (ผ่อนปรนเกณฑ์สอดคล้องภาพ-ข้อความ):** คัดเฉพาะคลาสข้อความตรงกัน (ภาพเป็นคลาสอะไรก็ได้) AND ความยาว **70 ตัวอักษรขึ้นไป** AND **ไม่ซ้ำซ้อน** กับชุดเดิม
- **Tier 4 (ผ่อนปรนสอดคล้องภาพ + ลดความยาว):** คัดคลาสข้อความตรงกัน AND ความยาว **50 ตัวอักษรขึ้นไป** AND **ไม่ซ้ำซ้อน** กับชุดเดิม
- **Tier 5 (ผ่อนปรนความซ้ำซ้อนเดิมแต่คุมคุณภาพ):** บังคับ Label ตรงกัน AND ความยาว **70 ตัวอักษรขึ้นไป** (อนุญาตให้ดึงข้อมูลที่เคยใช้ใน 500 รายการเดิมได้)
- **Tier 6:** บังคับ Label ตรงกัน AND ความยาว **50 ตัวอักษรขึ้นไป** (ดึงข้อมูลเดิมซ้ำได้)
- **Tier 7:** คลาสข้อความตรงกัน AND ความยาว **70 ตัวอักษรขึ้นไป** (ไม่ตรวจ Label ภาพ และดึงข้อมูลเดิมซ้ำได้)
- **Tier 8:** คลาสข้อความตรงกัน AND ความยาว **50 ตัวอักษรขึ้นไป** (ไม่ตรวจ Label ภาพ และดึงข้อมูลเดิมซ้ำได้)
- **Tier 9 (เกณฑ์ขั้นสุด):** คลาสข้อความตรงกัน AND ความยาว **30 ตัวอักษรขึ้นไป** (ไม่ตรวจ Label ภาพ และดึงข้อมูลเดิมซ้ำได้)

วิธีนี้ช่วยรับประกันว่า คลาสหลักที่ข้อมูลเยอะจะถูกดึงผ่าน **Tier 1** ที่สะอาดและเข้มข้นที่สุด ส่วนคลาสชนกลุ่มน้อย (เช่น คนหายหรือคนเจ็บ) จะได้รับการ "ยืม" ข้อมูลด้วยระดับการผ่อนเกณฑ์ที่เหมาะสมเพื่อเติมให้เต็มเป้าหมายการทดสอบ

---

## 3. การออกแบบคำสั่งสำหรับการแปลภาษา (Translation Prompt Design)

คำสั่งจะควบคุมให้โมเดลหลีกเลี่ยงการแปลแบบคำต่อคำ (Word-for-Word Translation) ซึ่งทำให้โครงสร้างภาษาไทยดูแปลกประหลาด หรือสำนวนเป็นภาษาหนังสือเกินไป โดยต้องปรับปรุงให้เป็นสำนวนภาษาคนไทยที่เขียนบนโซเชียลมีเดีย (เช่น Twitter/X หรือ Facebook) ในช่วงเกิดสถานการณ์ภัยพิบัติจริง

### 3.1 คำสั่งระบบ (System Instruction)
```markdown
You are an expert native Thai translator and disaster response analyst. Your task is to translate English disaster-related tweets into natural, fluent, and organic Thai as spoken by real Thai social media users (on platforms like Twitter/X and Facebook) during emergencies.

CRITICAL RULES FOR TRANSLATION:
1. DO NOT translate word-for-word (literal translation). Avoid English grammar structures in Thai (e.g., avoid excessive passive voice like "ถูกทำลายโดย..." unless natural).
2. Keep the translation conversational, casual, and natural. AVOID formal, academic, or overly journalistic/news-anchor Thai.
   - Do NOT use words like "เผย" (revealed), "ระบุว่า" (stated that), "ดำเนินงาน" (proceed), "ทำการ" (perform), "ส่งผลให้" (result in), or "ยังคง" (still) unless it fits a natural casual style.
   - Use natural connective and reporting verbs: e.g., use "บอกว่า" or "แจ้งว่า" instead of "เผย/ระบุ". Use "เพราะ" or "เนื่องจาก" instead of "ส่งผลให้".
   - DO NOT automatically append polite particles like "ครับ" (krub) or "ค่ะ" (kha) at the end of sentences. Most real tweets do not use them. Match the tone and level of formality of the original tweet.
   - AVOID overly conversational or playful particles like "นะ", "จ้า", "เนี่ย", "ดิ" or "เด้อ" unless the original tweet is clearly written in a playful/intimate personal chat style. Disaster-related social media posts (even casual ones) are informative and serious, so adding "นะ" at the end of reports sounds out of place and unnatural.
3. DO NOT change, add, or omit any crucial facts, numbers, datetime, or details (except locations, which must be localized to Thai places).
   - For example, if the English tweet mentions "10 injured", the Thai translation must report "บาดเจ็บ 10 คน" or "เจ็บ 10 คน" (avoid overly formal "บาดเจ็บ 10 ราย" if it sounds like an official police report).
   - DO NOT assume or invent details not present in the original text. For example, if the tweet says children are "missing", translate it as "สูญหาย" or "หายตัวไป". Do NOT assume or translate that they are "ติดอยู่ใต้ซาก" (trapped under rubble) unless the English text explicitly says "trapped under rubble/debris".
4. Preserve the context of the classification classes:
   - If the original mentions physical damage (houses, roads, bridges), ensure the Thai translation makes it very clear and descriptive of infrastructure damage (e.g., "ถนนขาด", "เสาไฟล้ม", "บ้านพัง").
   - If the original is a rescue/donation request, make sure the Thai translation sounds like a natural call for help or coordinate donation.
5. Localize English names of locations to appropriate Thai places (provinces, districts, streets) instead of transliterating them, to make the text sound completely organic to Thailand (e.g., change "Houston" or "California" to places like "เชียงราย", "อุบลราชธานี", "พะเยา", "สาย 304" depending on what fits the disaster type naturally).
```

### 3.2 ตัวอย่างการแปลสำหรับ Prompt (Few-Shot Examples)
เพื่อช่วยให้โมเดลเข้าใจแนวทาง "ไม่แปลตรงตัว แต่รักษาบริบทของ Class และการแทนที่ด้วยสถานที่ในไทย" จะมีตัวอย่างในคำสั่งผู้ใช้ดังนี้:

| ข้อความภาษาอังกฤษ (English) | แปลตรงตัว / แปลทางการหรือเล่นเกินไป (Bad Literal / Too Formal / Too Playful) | แปลธรรมชาติแบบไทยโซเชียลมีเดีย / รักษาบริบทและสถานที่ไทย (Good Natural & Localized) |
| :--- | :--- | :--- |
| *Please pray for Houston. My house is flooded and we need immediate rescue.* | โปรดสวดอ้อนวอนเพื่อฮิวสตัน บ้านของฉันถูกน้ำท่วมและเราต้องการการกู้ภัยทันที (แปลทื่อ)<br><br>ขอแรงใจให้เชียงรายด้วยครับ ตอนนี้บ้านผมน้ำท่วมสูงมาก ต้องการความช่วยเหลือด่วนครับ (ทางการ/สุภาพเกินไป) | ช่วยส่งใจ/ภาวนาให้เชียงรายด้วยนะ ตอนนี้บ้านน้ำท่วมสูงมาก อยากได้กู้ภัยเข้ามาช่วยด่วนเลย |
| *Red Cross volunteers are distributing food packs to 200 hurricane victims in Florida.* | อาสาสมัครสภากาชาดกำลังแจกจ่ายแพ็กเกจอาหารให้กับเหยื่อพายุเฮอริเคน 200 รายในฟลอริดา | อาสากาชาดกำลังเอาอาหารและถุงยังชีพไปแจกช่วยเหลือผู้ประสบภัยพายุ 200 คนแถวสุราษฎร์ |
| *Bridge on Route 9 collapsed due to the flash flood. Road closed.* | สะพานบนเส้นทาง 9 ทรุดตัวเนื่องจากน้ำท่วมฉับพลัน ถนนปิด (แปลทื่อ)<br><br>สะพานตรงถนนสาย 9 ขาดพังถล่มจากน้ำป่าไหลหลาก ตอนนี้ปิดการจราจรแล้วครับ (ทางการเกินไป) | สะพานตรงถนนสาย 9 ขาดเพราะน้ำป่าไหลหลาก ตอนนี้ปิดถนนไปแล้ว |
| *Relatives say children missing after a school collapsed in Mexico's deadly earthquake have sent WhatsApp messages.* | ญาติเผย เด็กๆ ที่ติดอยู่ใต้ซากอาคารเรียนหลังถล่มจากเหตุแผ่นดินไหวรุนแรง ยังคงส่งข้อความผ่าน WhatsApp ออกมาได้ครับ (ทางการ/สำนวนข่าวหนังสือพิมพ์เกินไป)<br><br>ญาติๆ บอกว่าเด็กที่ยังติดอยู่ใต้ซากโรงเรียนถล่มจากเหตุแผ่นดินไหว ส่งข้อความทาง WhatsApp ออกมาได้แล้วนะ (แปลเกินจริงเรื่องติดใต้ซาก และใช้คำลงท้าย "นะ" ที่ผิดกาลเทศะของการแจ้งข่าวภัยพิบัติ) | ญาติบอกว่าเด็กๆ ที่สูญหายหลังโรงเรียนถล่มเพราะแผ่นดินไหวรุนแรงที่เม็กซิโก ส่งข้อความผ่าน WhatsApp มาแล้ว |

### 3.3 คำสั่งผู้ใช้ (User Prompt Template)
```markdown
Translate the following English tweet into natural Thai according to the translation guidelines.

Tweet to translate:
"{tweet_text}"

Return only the translated Thai text. Do not add any introduction or explanation.
```

---

## 4. โครงสร้างข้อมูลและการบันทึกผล (Data Output Structure)
เมื่อประมวลผลแปลข้อมูลครบทั้ง 1,000 แถว ข้อมูลจะถูกจัดเก็บไว้ในไฟล์ CSV โครงสร้างดังนี้:

- **ที่จัดเก็บข้อมูลภาษาไทย (Output Path):** `e:/nlp-for-disaster/dataset/CrisisMMD_Thai_1000.csv`
- **โครงสร้างคอลัมน์:**
  1. `tweet_id`: ไอดีข้อความทวีต (ตรงกับต้นฉบับ)
  2. `original_english`: ข้อความภาษาอังกฤษต้นฉบับ
  3. `translated_thai`: ข้อความภาษาไทยที่ผ่านการแปลอย่างเป็นธรรมชาติ
  4. `true_text_info`: เฉลยความเกี่ยวข้อง (`informative` / `not_informative`)
  5. `true_text_human`: เฉลยหมวดหมู่ย่อย (`affected_individuals`, `infrastructure_and_utility_damage`, ... เพื่อนำไปทำนายและทดสอบประสิทธิภาพภาษาไทยต่อในอนาคต)
