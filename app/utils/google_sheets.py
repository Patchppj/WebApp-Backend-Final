import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
from pathlib import Path
import time

# กำหนดตำแหน่งของไฟล์ credentials
CREDENTIALS_PATH = Path(__file__).parent.parent / 'config' / 'google_sheets_credentials.json'

def create_headers(sheet):
    """
    สร้างหัวตารางใน Google Sheets
    
    Args:
        sheet: ออบเจ็กต์ worksheet ที่ต้องการสร้างหัวตาราง
    """
    # กำหนดหัวตาราง
    headers = [
        "วันที่และเวลา", "เพศ", "อายุ (ปี)", "น้ำหนัก (กก.)", "ส่วนสูง (ซม.)", 
        "ระยะเวลาที่อาศัยในชุมชน (ปี)", "จังหวัด", "อาชีพ", "การออกแรง/อาชีพที่ต้องออกแรง",
        "โรคตับคั่งไขมัน", "การสูบบุหรี่", "การดื่มแอลกอฮอล์", "โรคประจำตัวในครอบครัว",
        "การรับประทานอาหารทอด", "การรับประทานอาหารที่ปรุงด้วยกะทิ", "การรับประทานขนมหวาน/เครื่องดื่มสีสัน",
        "การดื่มชา/กาแฟ", "การออกกำลังกาย", "ระยะเวลาในการออกกำลังกาย", "ระยะเวลาในการนอนหลับต่อวัน (ชม.)",
        "ระดับน้ำตาลในเลือด", "ระดับ HbA1C", "ระดับอินซูลินในพลาสมา",
        "ระดับความเสี่ยง", "คะแนนความเสี่ยง (%)", "วันที่ทำนาย"
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
        sheet.format("A1:Z1", header_format)
        
        # ปรับความกว้างของคอลัมน์
        for i in range(len(headers)):
            col_letter = chr(65 + i) if i < 26 else chr(64 + i // 26) + chr(65 + i % 26)
            sheet.set_column_width(i+1, 150)  # ความกว้าง 150 พิกเซล
        
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
        if not os.path.exists(CREDENTIALS_PATH):
            print(f"ไม่พบไฟล์ credentials ที่: {CREDENTIALS_PATH}")
            raise FileNotFoundError(f"ไม่พบไฟล์ credentials ที่: {CREDENTIALS_PATH}")
        
        print(f"พบไฟล์ credentials ที่: {CREDENTIALS_PATH}")
            
        # กำหนดขอบเขตการเข้าถึง
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        # เชื่อมต่อกับ Google Sheets API
        print("กำลังเชื่อมต่อกับ Google Sheets API...")
        credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
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

def save_prediction_to_sheet(user_data, prediction_result, sheet_name="DiabetesPredictions"):
    """
    บันทึกข้อมูลการทำนายลงใน Google Sheets
    
    Args:
        user_data (UserData): ข้อมูลผู้ใช้ที่ใช้ในการทำนาย
        prediction_result (dict): ผลลัพธ์การทำนาย
        sheet_name (str): ชื่อของชีทที่ต้องการบันทึกข้อมูล
        
    Returns:
        bool: True ถ้าบันทึกสำเร็จ, False ถ้าไม่สำเร็จ
    """
    try:
        print(f"กำลังพยายามบันทึกข้อมูลลงใน Google Sheets: {sheet_name}")
        
        # เชื่อมต่อกับ Google Sheets
        sheet = get_google_sheet(sheet_name)
        if sheet is None:
            print(f"ไม่สามารถเชื่อมต่อกับ Google Sheets: {sheet_name}")
            return False
            
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
        row_data = [
            now,
            sex_text,
            user_data.age,
            user_data.weight,
            user_data.height,
            user_data.livingDuration,
            user_data.province,
            user_data.occupation,
            user_data.exertionOccupation,
            fatty_liver_text,
            smoking_text,
            alcohol_text,
            disease_family_text,
            fried_food_text,
            coconut_milk_text,
            candy_text,
            coffee_text,
            exercise_text,
            exercise_duration_text,
            user_data.sleepDuration,
            user_data.plasmaGlucose,
            user_data.HbA1C,
            user_data.plasmaInsulin,
            prediction_result["risk_level"],
            prediction_result["risk_percentage"],
            today
        ]
        
        # บันทึกข้อมูลลงในชีท
        sheet.append_row(row_data)
        
        return True
    except Exception as e:
        import traceback
        print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูลลง Google Sheets: {str(e)}")
        print("รายละเอียดข้อผิดพลาด:")
        traceback.print_exc()
        return False
