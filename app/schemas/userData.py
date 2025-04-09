from pydantic import BaseModel, Field

class UserData(BaseModel):
    # ข้อมูลส่วนตัว
    sex: int = Field(default=0, ge=0)
    age: int = Field(default=0, ge=0)
    weight: float = Field(default=0, ge=0)
    height: float = Field(default=0, ge=0)
    occupation: int = Field(default=0, ge=0)
    exertionOccupation: int = Field(default=0, ge=0)
    province: str = Field(default="")
    livingDuration: float = Field(default=0, ge=0)
    
    # โรคประจำตัว
    fattyLiver: int = Field(default=0, ge=0)
    coronaryAetery: int = Field(default=0, ge=0)
    chronicKidney: int = Field(default=0, ge=0)
    diseaseFamily: int = Field(default=0, ge=0)
    
    # พฤติกรรม
    smoking: int = Field(default=0, ge=0)
    alcohol: int = Field(default=0, ge=0)
    exercise: int = Field(default=0, ge=0)
    exerciseDuration: int = Field(default=0, ge=0)
    sleepDuration: float = Field(default=0, ge=0)
    socialTime: float = Field(default=0, ge=0)
    
    # พฤติกรรมการกิน
    eatSaltedFish: int = Field(default=0, ge=0)
    eatSausage: int = Field(default=0, ge=0)
    eatFriedFood: int = Field(default=0, ge=0)
    eatCoconutMilk: int = Field(default=0, ge=0)
    eatCandy: int = Field(default=0, ge=0)
    drinkCoffee: int = Field(default=0, ge=0)
    eatNut: int = Field(default=0, ge=0)
    eatFruit: int = Field(default=0, ge=0)
    eatInPlastic: int = Field(default=0, ge=0)
    drinkWater: float = Field(default=0, ge=0)
    
    # แบบประเมินคุณภาพชีวิต
    q1_health: int = Field(default=0, ge=0, le=5)
    q2_pain: int = Field(default=0, ge=0, le=5)
    q3_physical: int = Field(default=0, ge=0, le=5)
    q4_sleep: int = Field(default=0, ge=0, le=5)
    q5_feeling: int = Field(default=0, ge=0, le=5)
    q6_concentration: int = Field(default=0, ge=0, le=5)
    q7_self: int = Field(default=0, ge=0, le=5)
    q8_bodyImage: int = Field(default=0, ge=0, le=5)
    q9_negativeFeel: int = Field(default=0, ge=0, le=5)
    q10_dailyActivity: int = Field(default=0, ge=0, le=5)
    q11_onMedication: int = Field(default=0, ge=0, le=5)
    q12_working: int = Field(default=0, ge=0, le=5)
    q13_relationship: int = Field(default=0, ge=0, le=5)
    q14_socialSupport: int = Field(default=0, ge=0, le=5)
    q15_safety: int = Field(default=0, ge=0, le=5)
    q16_home: int = Field(default=0, ge=0, le=5)
    q17_financial: int = Field(default=0, ge=0, le=5)
    q18_healthService: int = Field(default=0, ge=0, le=5)
    q19_information: int = Field(default=0, ge=0, le=5)
    q20_leisure: int = Field(default=0, ge=0, le=5)
    q21_goodPhysical: int = Field(default=0, ge=0, le=5)
    q22_transportation: int = Field(default=0, ge=0, le=5)
    q23_spirituality: int = Field(default=0, ge=0, le=5)
    q24_mobility: int = Field(default=0, ge=0, le=5)
    q25_sex: int = Field(default=0, ge=0, le=5)
    q26_level: int = Field(default=0, ge=0, le=5)
    
    # ผลตรวจทางห้องปฏิบัติการ
    systolic: float = Field(default=0, ge=0)
    diastolic: float = Field(default=0, ge=0)
    plasmaGlucose: float = Field(default=0, ge=0)
    plasmaInsulin: float = Field(default=0, ge=0)
    HbA1C: float = Field(default=0, ge=0)
    HDL_cholesterol: float = Field(default=0, ge=0)
    LDL_cholesterol: float = Field(default=0, ge=0)
    triglyceride: float = Field(default=0, ge=0)
    creatinine: float = Field(default=0, ge=0)
    microalbumin: float = Field(default=0, ge=0)
    
    # ความพึงพอใจ
    f1_convenient: int = Field(default=0, ge=0, le=5)
    f2_question: int = Field(default=0, ge=0, le=5)
    f3_beautiful: int = Field(default=0, ge=0, le=5)
    f4_fast: int = Field(default=0, ge=0, le=5)