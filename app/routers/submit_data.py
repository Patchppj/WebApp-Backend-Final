from fastapi import APIRouter, HTTPException
from ..schemas.userData import UserData  
from ..google_sheets import write_to_google_sheet  # ไฟล์นี้จะสร้างต่อไป

router = APIRouter(prefix="/submit", tags=["user"])

@router.post("/data")
def submit_data(user_data: UserData):
    try:
        data_dict = user_data.dict()
        print("ข้อมูลที่ได้รับ:", data_dict)  # เพิ่ม Log
        write_to_google_sheet(data_dict)
        return {"status": "success", "message": "Data submitted to Google Sheets"}
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}") # เพิ่ม Log error ที่นี่ด้วย
        raise HTTPException(status_code=500, detail=str(e))
