import gspread
from google.oauth2.service_account import Credentials
from typing import Dict

# --- Configuration ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
SERVICE_ACCOUNT_FILE = 'C:/Users/PATCH/Downloads/credentials.json'  # <<<--- แก้ Path credentials.json ให้ถูกต้อง
SPREADSHEET_NAME = "UserDataStorage"  # <<<--- แก้ชื่อ Spreadsheet ให้ถูกต้อง
WORKSHEET_NAME = "Sheet1"  # <<<--- แก้ชื่อ Worksheet (Sheet ภายใน Spreadsheet) ให้ถูกต้อง

def get_sheets_client():
    """สร้างและคืนค่า gspread client."""
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    client = gspread.authorize(creds)
    return client


def write_to_google_sheet(data: Dict):
    try:
        print("กำลังพยายามเชื่อมต่อ Google Sheets...")
        client = get_sheets_client()
        print("เชื่อมต่อ Google Sheets สำเร็จ!")
        spreadsheet = client.open(SPREADSHEET_NAME)
        print(f"เปิด Spreadsheet: {SPREADSHEET_NAME} สำเร็จ!")
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        print(f"เลือก Worksheet: {WORKSHEET_NAME} สำเร็จ!")
        row = list(data.values())
        print(f"ข้อมูลที่จะเขียน: {row}")
        result = worksheet.append_row(row)
        print(f"เขียนข้อมูลสำเร็จ: {result}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดใน write_to_google_sheet: {e}")
        raise e