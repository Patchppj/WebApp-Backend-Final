import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
from pathlib import Path
import time
import uuid
import threading
from typing import Dict, Any, Optional

# กำหนดตำแหน่งของไฟล์ credentials
CREDENTIALS_PATH = Path(__file__).parent.parent / 'config' / 'google_sheets_credentials.json'

# ใช้ dictionary เพื่อเก็บข้อมูลชั่วคราวระหว่างรอการบันทึก
# key: session_id, value: {"diabetes": {...}, "hypertention": {...}, "user_data": {...}}
data_store = {}
data_lock = threading.Lock()  # ล็อคเพื่อป้องกัน race condition

def create_headers(sheet):
    """
    สร้างหัวตารางใน Google Sheets
    
    Args:
        sheet: ออบเจ็กต์ worksheet ที่ต้องการสร้างหัวตาราง
    """
    # กำหนดหัวตาราง
    headers = [
        "วันที่และเวลา", "เพศ", "อายุ (ปี)", "น้ำหนัก (กก.)", "ส่วนสูง (ซม.)",
        "อาชีพ", "อาชีพที่ต้องออกแรง", "จังหวัด", "ระยะเวลาที่อาศัยในชุมชน (ปี)",
        "โรคไขมันพอกตับ", "โรคหลอดเลือดหัวใจ", "โรคไตเรื้อรัง", "โรคประจำตัวในครอบครัว",
        "การสูบบุหรี่", "การดื่มแอลกอฮอล์", "การออกกำลังกาย", "ระยะเวลาในการออกกำลังกาย (นาที/วัน)",
        "ระยะเวลาในการนอนหลับต่อวัน (ชม.)", "ระยะเวลาการเล่นโซเชียลต่อวัน (ชม.)",
        "การกินปลาหมักดอง", "การกินไส้กรอก/แหนม/ไส้อั่ว", "การกินอาหารทอด", "การกินอาหารที่ปรุงด้วยกะทิ",
        "การกินขนมหวาน/เครื่องดื่มหวาน", "การดื่มชา/กาแฟ", "การกินถั่ว", "การกินผลไม้",
        "การกินอาหารในภาชนะพลาสติก", "การดื่มน้ำเปล่า", "โรคเรื้อรัง",
        "สุขภาพทั่วไป (Q1)", "ปวดเมื่อยร่างกาย (Q2)", "ความสามารถด้านร่างกาย (Q3)",
        "การนอนหลับ (Q4)", "ความรู้สึกสดชื่น (Q5)", "สมาธิ (Q6)", "การยอมรับตนเอง (Q7)",
        "ภาพลักษณ์ของตนเอง (Q8)", "อารมณ์ด้านลบ (Q9)", "การทำกิจวัตรประจำวัน (Q10)",
        "การใช้ยารักษาโรค (Q11)", "การทำงาน (Q12)", "ความสัมพันธ์กับคนรอบข้าง (Q13)",
        "การได้รับการสนับสนุนทางสังคม (Q14)", "ความปลอดภัยในชีวิต (Q15)",
        "สภาพแวดล้อมในบ้าน (Q16)", "สถานะทางการเงิน (Q17)", "การได้รับบริการสุขภาพ (Q18)",
        "การได้รับข้อมูลสุขภาพ (Q19)", "การมีกิจกรรมยามว่าง (Q20)", "สุขภาพร่างกายดี (Q21)",
        "การเดินทาง (Q22)", "จิตวิญญาณ (Q23)", "การเคลื่อนไหวร่างกาย (Q24)",
        "การมีเพศสัมพันธ์ (Q25)", "ระดับสุขภาพ (Q26)",
        "ความดันตัวบน (Systolic)", "ความดันตัวล่าง (Diastolic)",
        "ระดับน้ำตาลในเลือด (Plasma Glucose)", "ระดับอินซูลินในพลาสมา (Plasma Insulin)",
        "ระดับน้ำตาลสะสมในเลือด (HbA1C)", "ไขมันดี (HDL Cholesterol)", "ไขมันเลว (LDL Cholesterol)",
        "ไตรกลีเซอไรด์ (Triglyceride)", "ระดับครีเอตินิน (Creatinine)", "ไมโครอัลบูมินในปัสสาวะ (Microalbumin)",
        "ความสะดวกในการใช้งานแอป (F1)", "ความชัดเจนของคำถาม (F2)", "ความสวยงามของแอป (F3)", "ความรวดเร็วในการใช้งาน (F4)",
        "ระดับความเสี่ยงเบาหวาน", "คะแนนความเสี่ยงเบาหวาน (%)",
        "ระดับความเสี่ยงความดัน", "คะแนนความเสี่ยงความดัน (%)", "วันที่ทำนาย"
    ]
    # เพิ่มหัวตาราง
    sheet.append_row(headers)
    print("สร้างหัวตารางเรียบร้อยแล้ว")
    
    # จัดรูปแบบหัวตาราง (ตัวหนา พื้นหลังสี)
    try:
        # รอให้การเพิ่มหัวตารางเสร็จสมบูรณ์
        time.sleep(2)
        
        # จัดรูปแบบหัวตาราง
        header_format = {
            "backgroundColor": {"red": 0.8, "green": 0.8, "blue": 1.0},
            "horizontalAlignment": "CENTER",
            "textFormat": {"bold": True}
        }
        
        # กำหนดขนาดคอลัมน์
        # ใช้ช่วงที่ครอบคลุมทุกคอลัมน์
        # เนื่องจากมีมากกว่า 26 คอลัมน์ จึงต้องใช้ช่วงที่กว้างขึ้น
        # สร้างช่วงที่ครอบคลุมทุกคอลัมน์ในแถวแรก
        last_column = chr(65 + min(len(headers) - 1, 25))  # Z คือคอลัมน์ที่ 26
        
        if len(headers) <= 26:
            # ถ้ามีไม่เกิน 26 คอลัมน์ ใช้รูปแบบ A1:Z1
            format_range = f"A1:{last_column}1"
        else:
            # ถ้ามีมากกว่า 26 คอลัมน์ ใช้รูปแบบ A1:AA1, A1:AB1, หรือมากกว่า
            # คำนวณตัวอักษรสุดท้าย
            remaining = len(headers) - 26
            last_letter_1 = chr(64 + (remaining // 26) + 1)  # A, B, C, ...
            last_letter_2 = chr(64 + (remaining % 26) + 1)   # A, B, C, ...
            last_column = f"{last_letter_1}{last_letter_2}"
            format_range = f"A1:{last_column}1"
        
        print(f"[DEBUG] จัดรูปแบบหัวตารางในช่วง: {format_range}")
        sheet.format(format_range, header_format)
        
        # ปรับความกว้างของคอลัมน์ - ข้ามขั้นตอนนี้เนื่องจากไม่สนับสนุนใน gspread เวอร์ชันปัจจุบัน
        # สำหรับ gspread รุ่นใหม่ อาจใช้ sheet.set_column_width หรือ sheet.columns_auto_resize
        # แต่ในที่นี้เราจะข้ามการตั้งค่านี้ไป
        
        print("จัดรูปแบบหัวตารางเรียบร้อยแล้ว")
    except Exception as e:
        print(f"ไม่สามารถจัดรูปแบบหัวตารางได้: {str(e)}")
        # ไม่ต้องหยุดการทำงานหากไม่สามารถจัดรูปแบบได้

def get_google_sheet(sheet_name):
    """
    เชื่อมต่อกับ Google Sheets และเปิดชีทที่ต้องการ
    
    Args:
        sheet_name (str): ชื่อของชีทที่ต้องการเปิด
        
    Returns:
        worksheet: ออบเจ็กต์ worksheet ที่เปิดแล้ว
    """
    try:
        # ตรวจสอบว่ามีไฟล์ credentials หรือไม่
        # if not os.path.exists(CREDENTIALS_PATH):
        #     print(f"ไม่พบไฟล์ credentials ที่: {CREDENTIALS_PATH}")
        #     raise FileNotFoundError(f"ไม่พบไฟล์ credentials ที่: {CREDENTIALS_PATH}")
        
        # print(f"พบไฟล์ credentials ที่: {CREDENTIALS_PATH}")
            
        # กำหนดขอบเขตการเข้าถึง
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        # เชื่อมต่อกับ Google Sheets API
        print("กำลังเชื่อมต่อกับ Google Sheets API...")
        credentials = ServiceAccountCredentials.from_json_keyfile_name({
  "type": "service_account",
  "project_id": "web-application-457810",
  "private_key_id": "8bcaaff4b80a1e56b8b1191f9f7abf0c9dba4acd",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC8Wgh+endOIhQe\nbkT8tjcDpCVaSfU1PO/5XRSOOmx2Dq2b4DAr4YZu12LdHF70F4/bxun7hUk4kThA\nFo0rqJ4rBMnTRrGxniDbHQac2IrWdsz7lb4mlDEfMsi4QaLxJFNTfJF+jRXjXpT7\nzM73ILQPEj2j5dfnjws1m3Qcyo6Zr3jTa+IeyOX6erw9+OmhwpAOWksdq++ndbVI\ncbTq0uyZuFzg+b8gC2ieJNqqdxb0TVq9dwecroptJWqdgwzsxCfKgoON37NJjiZF\nVJrdWso954f/1OslaekBXakaY/XV2yjwEoYNi3e7s1WkusJo7wMPyAMYabeNqYS7\nZv290eQdAgMBAAECggEAHG7lpAeOGxgUkW4esUeQkSwmYgroHrFO81s/JHlisNpm\nZYbiaWgSMzQF6KXFGnQJc36uXm8wg9Q9pERt6zlOfJAB0l3JqqvZqeTSgBBFuo+/\nQftznmn16w/E3TaLXSZurcZlmwURWNGJDA9O7wowzwBMmIFfURrMRmYWkSzCz3rl\nv7TfJJF+FK1s5L9eqgm0yWUnKWVDTR2vi3drIAb85uHKXclnvjtmXQDpzUTur8D6\ndEF8TZO9CtEiicRnWghgneuFGqQIBsKifz5QGzOYyv9pghyjpvpppW4VLDp3rEhD\nq/asaM+h0IMURNF624ZLc5UhwCTWoZSEp74bs+9cwQKBgQD1GaGxJZpXJu/JfZb2\no+i2pr1p9IScRnJLbcY7RlDX3qx6X5ku9yCmS9dlixGr/wH1WkPgSYHrGvLXsw8G\npPETQIlsnh+MPCCBSA8jTiHpvX+tVxgywIp/4ZVjEYeJSzHWy3ao6hTKWgjoYp3o\nRnNd/7XKBNyGzCl36oEH8we1twKBgQDEulfQihIywU3owkX720VixyGgVfW6rvX5\nseqC1ZVX2zxK2TBk548kgHvtX1NORRH/Z3Hufa3L7ygnZ051RwDmAZXsaNIpiId8\n609IQOCWkSsD4iBZWkvd4XrarPVwpSj6H8l3ByMMDKpw6wVC9VNOAHYjjIy9N57N\nStLTg5GUywKBgHufHPW7cJgqlGD9TkpCK9sMSJcLdGNaCMXQrV4yPg5TSo1CcIHG\ntZoKwK5/sT8eFL/KclfK25NYeUmgD6EbSliagXXeXy0dGov6a2A0RViecqpcNmFK\nydBsWCuqqMDvw0iCQOw7fJb/SGTlcJ26AvBTTD6DqzL2AKhyB/iZdLcLAoGAMcso\npvQLnyUmXx+tLw5VBad5b2fShqn6QHUz8mG1J0OqgxduFFw38vlCZNaX81uwLoE0\naTUOZGvoMfEH/s81/wGvvOLbLwALqya0LomdTv73cEgv/+3G/iYwPmFAzn4/XO/m\nwmXgDRC3o1UZQ9VsfHXJcT4F8W6+lx+1NSw8EPkCgYA7YFKVRUyxQ+bNZ17Yjwmd\nXr+mwpajy2RQZ1plTVhpRCifJX4BBDx1YBbHK68323uPOxYVdMfyo65CBDM6TrzP\nYQTpOf0jwjYHxwc6y7ib3pv29AZGp3AHWEqGSM96eEcT/P8CZipv63OyRlz78s1U\nrJaYgUcxEuh/Yo/Azkjl6Q==\n-----END PRIVATE KEY-----\n",
  "client_email": "direbetes@web-application-457810.iam.gserviceaccount.com",
  "client_id": "108481939863532815382",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/direbetes%40web-application-457810.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
, scope)
        client = gspread.authorize(credentials)
        
        # เปิดชีทที่ต้องการ
        print(f"กำลังพยายามเปิดชีท: {sheet_name}")
        sheet = client.open(sheet_name).sheet1
        print(f"เปิดชีท: {sheet_name} สำเร็จ")
        
        return sheet
    except FileNotFoundError as e:
        print(f"ข้อผิดพลาด: {str(e)}")
        return None
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Google Sheets: {str(e)}")
        return None

def generate_session_id():
    """
    สร้าง session ID สำหรับเชื่อมโยงข้อมูลจากทั้งสองเส้น
    
    Returns:
        str: session ID
    """
    return str(uuid.uuid4())

def store_prediction_data(session_id, prediction_type, user_data, prediction_result):
    """
    เก็บข้อมูลการทำนายชั่วคราวก่อนบันทึกลง Google Sheets
    
    Args:
        session_id (str): ID สำหรับเชื่อมโยงข้อมูล
        prediction_type (str): ประเภทของการทำนาย ("diabetes" หรือ "hypertention")
        user_data: ข้อมูลผู้ใช้
        prediction_result: ผลลัพธ์การทำนาย
        
    Returns:
        bool: True ถ้าข้อมูลพร้อมบันทึก (มีข้อมูลครบทั้งสองประเภท), False ถ้ายังไม่พร้อม
    """
    with data_lock:
        # สร้าง entry ใหม่ถ้ายังไม่มี
        if session_id not in data_store:
            data_store[session_id] = {"user_data": user_data}
        
        # เก็บข้อมูลการทำนาย
        data_store[session_id][prediction_type] = prediction_result
        
        # ตรวจสอบว่ามีข้อมูลครบทั้งสองประเภทหรือไม่
        has_diabetes = "diabetes" in data_store[session_id]
        has_hypertention = "hypertention" in data_store[session_id]
        
        return has_diabetes and has_hypertention

def save_combined_data_to_sheet(session_id, sheet_name="DiabetesPredictions"):
    """
    บันทึกข้อมูลการทำนายทั้งสองประเภทลงใน Google Sheets
    
    Args:
        session_id (str): ID สำหรับเชื่อมโยงข้อมูล
        sheet_name (str): ชื่อของชีทที่ต้องการบันทึกข้อมูล
        
    Returns:
        bool: True ถ้าบันทึกสำเร็จ, False ถ้าไม่สำเร็จ
    """
    try:
        print(f"\n[DEBUG] กำลังบันทึกข้อมูลลง Google Sheets '{sheet_name}' สำหรับ session ID: {session_id}")
        print(f"[DEBUG] ข้อมูลที่มีอยู่ในระบบ: {list(data_store.keys())}")
        
        with data_lock:
            # ตรวจสอบว่ามีข้อมูลสำหรับ session_id นี้หรือไม่
            if session_id not in data_store:
                print(f"[ERROR] ไม่พบข้อมูลสำหรับ session ID: {session_id}")
                return False
            
            # ตรวจสอบว่ามีข้อมูลครบทั้งสองประเภทหรือไม่
            data = data_store[session_id]
            print(f"[DEBUG] ข้อมูลที่มีสำหรับ session ID {session_id}: {list(data.keys())}")
            
            if "diabetes" not in data or "hypertention" not in data:
                print(f"[ERROR] ข้อมูลไม่ครบสำหรับ session ID: {session_id}")
                print(f"[DEBUG] ข้อมูลที่มี: {list(data.keys())}")
                return False
            
            user_data = data["user_data"]
            diabetes_result = data["diabetes"]
            hypertention_result = data["hypertention"]
            print(f"[DEBUG] ดึงข้อมูลสำเร็จ: user_data, diabetes_result, hypertention_result")
        
        # เชื่อมต่อกับ Google Sheets
        print(f"[DEBUG] กำลังเชื่อมต่อกับ Google Sheets: {sheet_name}")
        sheet = get_google_sheet(sheet_name)
        if sheet is None:
            print(f"[ERROR] ไม่สามารถเชื่อมต่อกับ Google Sheets: {sheet_name}")
            return False
        print(f"[DEBUG] เชื่อมต่อกับ Google Sheets สำเร็จ")
            
        # ตรวจสอบว่ามีหัวตารางหรือไม่ ถ้าไม่มีให้สร้าง
        try:
            # ลองดึงข้อมูลแถวแรกเพื่อตรวจสอบว่ามีหัวตารางหรือไม่
            first_row = sheet.row_values(1)
            if not first_row or len(first_row) < 5:  # ถ้าไม่มีข้อมูลหรือข้อมูลไม่ครบ
                print("ไม่พบหัวตาราง กำลังสร้างหัวตาราง...")
                # ล้างข้อมูลทั้งหมดและสร้างหัวตารางใหม่
                sheet.clear()
                create_headers(sheet)
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการตรวจสอบหัวตาราง: {str(e)}")
            print("กำลังสร้างหัวตารางใหม่...")
            create_headers(sheet)
            
        # แปลงข้อมูลเพศเป็นข้อความ
        sex_text = "ชาย" if user_data.sex == 1 else "หญิง" if user_data.sex == 2 else "ไม่ระบุ"
        
        # แปลงข้อมูลอื่นๆ เป็นข้อความที่อ่านง่าย
        fatty_liver_text = "มี" if user_data.fattyLiver == 1 else "ไม่มี" if user_data.fattyLiver == 2 else "ไม่ระบุ"
        smoking_text = "สูบ" if user_data.smoking == 1 else "ไม่สูบ" if user_data.smoking == 2 else "ไม่ระบุ"
        alcohol_text = "ดื่ม" if user_data.alcohol == 1 else "ไม่ดื่ม" if user_data.alcohol == 2 else "ไม่ระบุ"
        disease_family_text = "มี" if user_data.diseaseFamily == 1 else "ไม่มี" if user_data.diseaseFamily == 2 else "ไม่ระบุ"
        fried_food_text = "ทาน" if user_data.eatFriedFood == 1 else "ไม่ทาน" if user_data.eatFriedFood == 2 else "ไม่ระบุ"
        coconut_milk_text = "ทาน" if user_data.eatCoconutMilk == 1 else "ไม่ทาน" if user_data.eatCoconutMilk == 2 else "ไม่ระบุ"
        candy_text = "ทาน" if user_data.eatCandy == 1 else "ไม่ทาน" if user_data.eatCandy == 2 else "ไม่ระบุ"
        coffee_text = "ดื่ม" if user_data.drinkCoffee == 1 else "ไม่ดื่ม" if user_data.drinkCoffee == 2 else "ไม่ระบุ"
        
        # แปลงข้อมูลการออกกำลังกาย
        exercise_text = "ไม่ได้ออกกำลังกาย" if user_data.exercise == 1 else \
                        "ออกกำลังกายน้อยกว่า 1 ปี" if user_data.exercise == 2 else \
                        "ออกกำลังกาย 1-5 ปี" if user_data.exercise == 3 else \
                        "ออกกำลังกาย 6-10 ปี" if user_data.exercise == 4 else \
                        "ออกกำลังกายมากกว่า 10 ปี" if user_data.exercise == 5 else "ไม่ระบุ"
        
        exercise_duration_text = "ไม่ได้ออกกำลังกาย" if user_data.exerciseDuration == 1 else \
                                "น้อยกว่า 30 นาที/ครั้ง" if user_data.exerciseDuration == 2 else \
                                "30-60 นาที/ครั้ง" if user_data.exerciseDuration == 3 else \
                                "มากกว่า 60 นาที/ครั้ง" if user_data.exerciseDuration == 4 else "ไม่ระบุ"
        
        # สร้างข้อมูลที่จะบันทึก
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today = datetime.now().strftime("%Y-%m-%d")
        
        # แปลงข้อมูลเพิ่มเติม
        coronary_aetery_text = "มี" if hasattr(user_data, 'coronaryAetery') and user_data.coronaryAetery == 1 else "ไม่มี" if hasattr(user_data, 'coronaryAetery') and user_data.coronaryAetery == 2 else "ไม่ระบุ"
        chronic_kidney_text = "มี" if hasattr(user_data, 'chronicKidney') and user_data.chronicKidney == 1 else "ไม่มี" if hasattr(user_data, 'chronicKidney') and user_data.chronicKidney == 2 else "ไม่ระบุ"
        salted_fish_text = "ทาน" if hasattr(user_data, 'eatSaltedFish') and user_data.eatSaltedFish == 1 else "ไม่ทาน" if hasattr(user_data, 'eatSaltedFish') and user_data.eatSaltedFish == 2 else "ไม่ระบุ"
        sausage_text = "ทาน" if hasattr(user_data, 'eatSausage') and user_data.eatSausage == 1 else "ไม่ทาน" if hasattr(user_data, 'eatSausage') and user_data.eatSausage == 2 else "ไม่ระบุ"
        nut_text = "ทาน" if hasattr(user_data, 'eatNut') and user_data.eatNut == 1 else "ไม่ทาน" if hasattr(user_data, 'eatNut') and user_data.eatNut == 2 else "ไม่ระบุ"
        fruit_text = "ทาน" if hasattr(user_data, 'eatFruit') and user_data.eatFruit == 1 else "ไม่ทาน" if hasattr(user_data, 'eatFruit') and user_data.eatFruit == 2 else "ไม่ระบุ"
        plastic_text = "ทาน" if hasattr(user_data, 'eatInPlastic') and user_data.eatInPlastic == 1 else "ไม่ทาน" if hasattr(user_data, 'eatInPlastic') and user_data.eatInPlastic == 2 else "ไม่ระบุ"
        
        # สร้างข้อมูลที่จะบันทึกตามหัวตารางใหม่
        row_data = [
            now,                                           # วันที่และเวลา
            sex_text,                                      # เพศ
            user_data.age,                                # อายุ (ปี)
            user_data.weight,                             # น้ำหนัก (กก.)
            user_data.height,                             # ส่วนสูง (ซม.)
            user_data.occupation,                         # อาชีพ
            user_data.exertionOccupation,                 # อาชีพที่ต้องออกแรง
            user_data.province,                           # จังหวัด
            user_data.livingDuration,                     # ระยะเวลาที่อาศัยในชุมชน (ปี)
            fatty_liver_text,                             # โรคไขมันพอกตับ
            coronary_aetery_text,                         # โรคหลอดเลือดหัวใจ
            chronic_kidney_text,                          # โรคไตเรื้อรัง
            disease_family_text,                          # โรคประจำตัวในครอบครัว
            smoking_text,                                 # การสูบบุหรี่
            alcohol_text,                                 # การดื่มแอลกอฮอล์
            exercise_text,                                # การออกกำลังกาย
            exercise_duration_text,                       # ระยะเวลาในการออกกำลังกาย (นาที/วัน)
            user_data.sleepDuration,                      # ระยะเวลาในการนอนหลับต่อวัน (ชม.)
            user_data.socialTime,                         # ระยะเวลาการเล่นโซเชียลต่อวัน (ชม.)
            salted_fish_text,                             # การกินปลาหมักดอง
            sausage_text,                                 # การกินไส้กรอก/แหนม/ไส้อั่ว
            fried_food_text,                              # การกินอาหารทอด
            coconut_milk_text,                            # การกินอาหารที่ปรุงด้วยกะทิ
            candy_text,                                   # การกินขนมหวาน/เครื่องดื่มหวาน
            coffee_text,                                  # การดื่มชา/กาแฟ
            nut_text,                                     # การกินถั่ว
            fruit_text,                                   # การกินผลไม้
            plastic_text,                                 # การกินอาหารในภาชนะพลาสติก
            getattr(user_data, 'drinkWater', 0),          # การดื่มน้ำเปล่า
            getattr(user_data, 'chronicDisease', ''),     # โรคเรื้อรัง
            
            # แบบประเมินคุณภาพชีวิต (Q1-Q26)
            getattr(user_data, 'q1_health', 0),
            getattr(user_data, 'q2_pain', 0),
            getattr(user_data, 'q3_physical', 0),
            getattr(user_data, 'q4_sleep', 0),
            getattr(user_data, 'q5_feeling', 0),
            getattr(user_data, 'q6_concentration', 0),
            getattr(user_data, 'q7_self', 0),
            getattr(user_data, 'q8_bodyImage', 0),
            getattr(user_data, 'q9_negativeFeel', 0),
            getattr(user_data, 'q10_dailyActivity', 0),
            getattr(user_data, 'q11_onMedication', 0),
            getattr(user_data, 'q12_working', 0),
            getattr(user_data, 'q13_relationship', 0),
            getattr(user_data, 'q14_socialSupport', 0),
            getattr(user_data, 'q15_safety', 0),
            getattr(user_data, 'q16_home', 0),
            getattr(user_data, 'q17_financial', 0),
            getattr(user_data, 'q18_healthService', 0),
            getattr(user_data, 'q19_information', 0),
            getattr(user_data, 'q20_leisure', 0),
            getattr(user_data, 'q21_goodPhysical', 0),
            getattr(user_data, 'q22_transportation', 0),
            getattr(user_data, 'q23_spirituality', 0),
            getattr(user_data, 'q24_mobility', 0),
            getattr(user_data, 'q25_sex', 0),
            getattr(user_data, 'q26_level', 0),
            
            # ข้อมูลทางการแพทย์
            getattr(user_data, 'systolic', 0),
            getattr(user_data, 'diastolic', 0),
            user_data.plasmaGlucose,
            user_data.plasmaInsulin,
            user_data.HbA1C,
            getattr(user_data, 'HDL_cholesterol', 0),
            getattr(user_data, 'LDL_cholesterol', 0),
            getattr(user_data, 'triglyceride', 0),
            user_data.creatinine,
            getattr(user_data, 'microalbumin', 0),
            
            # ความพึงพอใจ
            getattr(user_data, 'f1_convenient', 0),
            getattr(user_data, 'f2_question', 0),
            getattr(user_data, 'f3_beautiful', 0),
            getattr(user_data, 'f4_fast', 0),
            
            # ผลการทำนาย
            diabetes_result["risk_level"],
            diabetes_result["risk_percentage"],
            hypertention_result["risk_level"],
            hypertention_result["risk_percentage"],
            today
        ]
        
        # บันทึกข้อมูลลงในชีท
        sheet.append_row(row_data)
        
        # ลบข้อมูลออกจาก data_store เมื่อบันทึกเสร็จแล้ว
        with data_lock:
            if session_id in data_store:
                del data_store[session_id]
        
        return True
    except Exception as e:
        import traceback
        print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูลลง Google Sheets: {str(e)}")
        print("รายละเอียดข้อผิดพลาด:")
        traceback.print_exc()
        return False
