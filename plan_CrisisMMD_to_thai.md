# แผนการแปลชุดข้อมูลภัยพิบัติสู่ภาษาไทย (CrisisMMD Translation to Thai - Plan)

เอกสารฉบับนี้กำหนดแผนและแนวทางการจัดเตรียมชุดข้อมูลภัยพิบัติแปลไทย **plan_CrisisMMD_to_thai.md** โดยทำการสุ่มตัวอย่างข้อความทวีตใหม่จากชุดข้อมูลอ้างอิงสากล `CrisisMMD_v2.0` นำมาแปลภาษาให้มีความเป็นธรรมชาติ เข้ากับบริบทคำพูดของคนไทย แต่ยังคงความหมายและโครงสร้างดั้งเดิมอย่างครบถ้วนสำหรับใช้ในภารกิจจัดหมวดหมู่ภัยพิบัติ

---

## 1. วัตถุประสงค์ (Objectives)
- จัดทำชุดข้อมูลข้อความภัยพิบัติภาษาไทยมาตรฐานจำนวน **500 รายการ** ที่แปลมาจากต้นฉบับภาษาอังกฤษของ CrisisMMD_v2.0
- ใช้โมเดล **`gemini-3.1-lite-flash`** (หรือรุ่นที่เหมาะสมในการแปลความเร็วสูง) เพื่อทำการแปลงภาษาแบบรักษาบริบท
- ออกแบบคำสั่ง Prompt ในการแปลโดยเน้นให้โมเดลรังสรรค์สำนวนการเขียนเหมือนคนไทยเขียนเองโดยธรรมชาติ บนแพลตฟอร์มโซเชียลมีเดีย แต่ต้องไม่สูญเสียความหมายหรือสถิติข้อมูลที่เป็นเกณฑ์การจัดหมวดหมู่ภัยพิบัติ (Informativeness & Category Context)

---

## 2. ขั้นตอนการสุ่มข้อมูลตัวอย่าง (Sampling & Exclusion Strategy)
เพื่อไม่ให้ชุดข้อมูลซ้ำซ้อนกับชุดเดิมที่ใช้ในการทดลอง Exp 01 และ Exp 02 สคริปต์เตรียมข้อมูลจะดำเนินการดังนี้:
1. โหลดข้อมูลแถวข้อความทวีตภาษาอังกฤษทั้งหมดจาก TSV ของ CrisisMMD
2. ทำการหักลบ (Exclude) บรรทัดข้อมูลทวีตจำนวน 500 แถวแรกที่เคยถูกดึงไปใช้ประเมินผลใน Exp 01 ออกไปจากบัญชีรายชื่อทั้งหมด
3. สุ่มดึงข้อความรอบใหม่จำนวน **500 แถว** (สุ่มกระจายสัดส่วนตามประเภทภัยพิบัติ) เพื่อใช้เป็นชุดข้อความตั้งต้นในการแปลภาษาไทย

---

## 3. การออกแบบคำสั่งสำหรับการแปลภาษา (Translation Prompt Design)

คำสั่งจะควบคุมให้โมเดลหลีกเลี่ยงการแปลแบบคำต่อคำ (Word-for-Word Translation) ซึ่งทำให้โครงสร้างภาษาไทยดูแปลกประหลาด หรือสำนวนเป็นภาษาหนังสือเกินไป โดยต้องปรับปรุงให้เป็นสำนวนภาษาคนไทยที่เขียนบนโซเชียลมีเดีย (เช่น Twitter/X หรือ Facebook) ในช่วงเกิดสถานการณ์ภัยพิบัติจริง

### 3.1 คำสั่งระบบ (System Instruction)
```markdown
You are an expert native Thai translator and disaster response analyst. Your task is to translate English disaster-related tweets into natural, fluent, and organic Thai as spoken by real Thai social media users during emergencies.

CRITICAL RULES FOR TRANSLATION:
1. DO NOT translate word-for-word (literal translation). Avoid English grammar structures in Thai (e.g., avoid excessive passive voice like "ถูกทำลายโดย..." unless natural).
2. Keep the translation conversational, natural, and like a native Thai social media poster (e.g., using terms like "แจ้งข่าวครับ", "ส่งใจช่วยนะครับ", "หน่วยกู้ภัย", "น้ำท่วมสูง", "เสียหายหนัก").
3. DO NOT change, add, or omit any crucial facts, numbers, datetime, or details (except locations, which must be localized to Thai places). For example, if the English tweet mentions "10 injured", the Thai translation must report "บาดเจ็บ 10 ราย" or "เจ็บ 10 คน".
4. Preserve the context of the classification classes:
   - If the original mentions physical damage (houses, roads, bridges), ensure the Thai translation makes it very clear and descriptive of infrastructure damage (e.g., "ถนนขาด", "เสาไฟล้ม", "บ้านพัง").
   - If the original is a rescue/donation request, make sure the Thai translation sounds like a natural call for help or coordinate donation.
5. Localize English names of locations to appropriate Thai places (provinces, districts, streets) instead of transliterating them, to make the text sound completely organic to Thailand (e.g., change "Houston" or "California" to places like "เชียงราย", "อุบลราชธานี", "พะเยา", "สาย 304" depending on what fits the disaster type naturally).
```

### 3.2 ตัวอย่างการแปลสำหรับ Prompt (Few-Shot Examples)
เพื่อช่วยให้โมเดลเข้าใจแนวทาง "ไม่แปลตรงตัว แต่รักษาบริบทของ Class และการแทนที่ด้วยสถานที่ในไทย" จะมีตัวอย่างในคำสั่งผู้ใช้ดังนี้:

| ข้อความภาษาอังกฤษ (English) | แปลตรงตัว / แปลทื่อ (Bad Literal) | แปลธรรมชาติแบบไทย / รักษาคลาสและสถานที่ไทย (Good Natural & Localized) |
| :--- | :--- | :--- |
| *Please pray for Houston. My house is flooded and we need immediate rescue.* | โปรดสวดอ้อนวอนเพื่อฮิวสตัน บ้านของฉันถูกน้ำท่วมและเราต้องการการกู้ภัยทันที | ขอแรงใจให้เชียงรายด้วยครับ ตอนนี้บ้านผมน้ำท่วมสูงมาก ต้องการความช่วยเหลือด่วนครับ |
| *Red Cross volunteers are distributing food packs to 200 hurricane victims in Florida.* | อาสาสมัครสภากาชาดกำลังแจกจ่ายแพ็กเกจอาหารให้กับเหยื่อพายุเฮอริเคน 200 รายในฟลอริดา | อาสาสมัครกาชาดกำลังนำถุงยังชีพและอาหารไปแจกจ่ายช่วยเหลือผู้ประสบภัยพายุ 200 คนในพื้นที่สุราษฎร์ธานี |
| *Bridge on Route 9 collapsed due to the flash flood. Road closed.* | สะพานบนเส้นทาง 9 ทรุดตัวเนื่องจากน้ำท่วมฉับพลัน ถนนปิด | สะพานตรงถนนสาย 9 ขาดพังถล่มจากน้ำป่าไหลหลาก ตอนนี้ปิดการจราจรแล้วครับ |

### 3.3 คำสั่งผู้ใช้ (User Prompt Template)
```markdown
Translate the following English tweet into natural Thai according to the translation guidelines.

Tweet to translate:
"{tweet_text}"

Return only the translated Thai text. Do not add any introduction or explanation.
```

---

## 4. โครงสร้างข้อมูลและการบันทึกผล (Data Output Structure)
เมื่อประมวลผลแปลข้อมูลครบทั้ง 500 แถว ข้อมูลจะถูกจัดเก็บไว้ในไฟล์ CSV โครงสร้างดังนี้:

- **ที่จัดเก็บข้อมูลภาษาไทย (Output Path):** `e:/nlp-for-disaster/data/CrisisMMD_Thai_500.csv`
- **โครงสร้างคอลัมน์:**
  1. `tweet_id`: ไอดีข้อความทวีต (ตรงกับต้นฉบับ)
  2. `original_english`: ข้อความภาษาอังกฤษต้นฉบับ
  3. `translated_thai`: ข้อความภาษาไทยที่ผ่านการแปลอย่างเป็นธรรมชาติ
  4. `true_text_info`: เฉลยความเกี่ยวข้อง (`informative` / `not_informative`)
  5. `true_text_human`: เฉลยหมวดหมู่ย่อย (`affected_individuals`, `infrastructure_and_utility_damage`, ... เพื่อนำไปทำนายและทดสอบประสิทธิภาพภาษาไทยต่อในอนาคต)
