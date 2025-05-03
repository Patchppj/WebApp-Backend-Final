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
        {
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
}, scopes=SCOPES
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