from math import log
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pickle
import numpy as np
from pathlib import Path
import xgboost as xgb  # เพิ่ม import xgboost
from ..schemas.userData import UserData

router = APIRouter(
    prefix="/hypertention",
    tags=["hypertention"]
)

# โหลดโมเดล XGBoost
# แก้ไขเส้นทางไฟล์โมเดล
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgb_hypertention_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    raise Exception(f"ไม่สามารถโหลดโมเดลได้: {str(e)}")

@router.post("/predict")
async def predict_risk(data: UserData):
    try:
        print("Received data:", data)

        feature_array = np.zeros(88)  # ปรับขนาดเป็น 88 ตามจำนวน features

        # ตัวแปรต่อเนื่อง
        feature_array[0] = data.height  # Height
        feature_array[1] = data.weight  # Weight
        feature_array[2] = data.livingDuration  # Duration of living

        # Sex
        sex = data.sex
        if sex == 1:
            feature_array[3] = 1  # Sex_1
        elif sex == 2:
            feature_array[4] = 1  # Sex_2

        # Age
        age = data.age
        if age == 1:
            feature_array[86] = 1  # Age_1
        elif age == 2:
            feature_array[5] = 1  # Age_2:
        elif age == 3:
            feature_array[6] = 1  # Age_3

        # Systolic
        systolic = data.systolic
        if systolic == 1:
            feature_array[87] = 1  # Systolic_1
        elif systolic == 2:
            feature_array[7] = 1  # Systolic_2
        elif systolic == 3:
            feature_array[8] = 1  # Systolic_3

        # Diastolic
        diastolic = data.diastolic
        if diastolic == 1:
            feature_array[9] = 1  # Diastolic_1
        elif diastolic == 2:
            feature_array[10] = 1  # Diastolic_2
        elif diastolic == 3:
            feature_array[11] = 1  # Diastolic_3

        # Occupation
        occupation = data.occupation
        if occupation == 1:
            feature_array[12] = 1  # Occupation_1
        elif occupation == 2:
            feature_array[13] = 1  # Occupation_2
        elif occupation == 3:
            feature_array[14] = 1  # Occupation_3
        elif occupation == 4:
            feature_array[15] = 1  # Occupation_4

        # Average time spent on social media per day (per hour)
        socialMedia = data.socialTime
        if socialMedia == 1:
            feature_array[16] = 1  # SocialMedia_1
        elif socialMedia == 2:
            feature_array[17] = 1  # SocialMedia_2
        elif socialMedia == 3:
            feature_array[18] = 1  # SocialMedia_3

        # Coronary artery disease
        coronaryAetery = data.coronaryAetery
        if coronaryAetery == 1:
            feature_array[19] = 1  # CoronaryAetery_1
        elif coronaryAetery == 2:
            feature_array[20] = 1  # CoronaryAetery_2

        # Fatty liver disease
        fattyLiver = data.fattyLiver
        if fattyLiver == 1:
            feature_array[21] = 1  # FattyLiver_1
        elif fattyLiver == 2:
            feature_array[22] = 1  # FattyLiver_2

        # Chronic kidney disease
        chronicKidney = data.chronicKidney
        if chronicKidney == 1:
            feature_array[23] = 1  # ChronicKidney_1
        elif chronicKidney == 2:
            feature_array[24] = 1  # ChronicKidney_2

        # Smoking
        smoking = data.smoking
        if smoking == 1:
            feature_array[25] = 1  # Smoking_1
        elif smoking == 2:
            feature_array[26] = 1  # Smoking_2

        # Drinking alcohol
        drinkingAlcohol = data.alcohol
        if drinkingAlcohol == 1:
            feature_array[27] = 1  # DrinkingAlcohol_1
        elif drinkingAlcohol == 2:
            feature_array[28] = 1  # DrinkingAlcohol_2

        # Underlying disease among family
        familyDisease = data.diseaseFamily
        if familyDisease == 1:
            feature_array[29] = 1  # FamilyDisease_1
        elif familyDisease == 2:
            feature_array[30] = 1  # FamilyDisease_2
        elif familyDisease == 3:
            feature_array[31] = 1  # FamilyDisease_3

        # Eating salted fish, salted meat
        eatingSalted = data.eatSaltedFish
        if eatingSalted == 1:
            feature_array[32] = 1  # EatingSalted_1
        elif eatingSalted == 2:
            feature_array[33] = 1  # EatingSalted_2

       # Eating fried foods
        eatingFried = data.eatFriedFood
        if eatingFried == 1:
            feature_array[34] = 1  # EatingFried_1
        elif eatingFried == 2:
            feature_array[35] = 1  # EatingFried_2

        # Eating food cooked with coconut milk
        eatingCoconut = data.eatCoconutMilk
        if eatingCoconut == 1:
            feature_array[36] = 1  # EatingCoconut_1
        elif eatingCoconut == 2:
            feature_array[37] = 1  # EatingCoconut_2

        # Consumption of tea, coffee (glass)
        drinkingTeaCoffee = data.drinkCoffee
        if drinkingTeaCoffee == 1:
            feature_array[38] = 1  # DrinkingTeaCoffee_1
        elif drinkingTeaCoffee == 2:
            feature_array[39] = 1  # DrinkingTeaCoffee_2

        # Eating Place food packed in plastic bags/foam boxes in the microwave
        eatingPlastic = data.eatInPlastic
        if eatingPlastic == 1:
            feature_array[40] = 1  # EatingPlastic_1
        elif eatingPlastic == 2:
            feature_array[41] = 1  # EatingPlastic_2

        # Drinking water per day
        drinkingWater = data.drinkWater
        if drinkingWater == 1:
            feature_array[42] = 1  # DrinkingWater_1
        elif drinkingWater == 2:
            feature_array[43] = 1  # DrinkingWater_2
        elif drinkingWater == 3:
            feature_array[44] = 1  # DrinkingWater_3
        elif drinkingWater == 4:
            feature_array[45] = 1  # DrinkingWater_4

        # Exercise (year)
        exercise = data.exercise
        if exercise == 1:
            feature_array[46] = 1  # Exercise_1
        elif exercise == 2:
            feature_array[47] = 1  # Exercise_2
        elif exercise == 3:
            feature_array[48] = 1  # Exercise_3
        elif exercise == 4:
            feature_array[49] = 1  # Exercise_4
        elif exercise == 5:
            feature_array[50] = 1  # Exercise_5

        # Duration of exercise
        exerciseDuration = data.exerciseDuration
        if exerciseDuration == 1:
            feature_array[51] = 1  # ExerciseDuration_1
        elif exerciseDuration == 2:
            feature_array[52] = 1  # ExerciseDuration_2
        elif exerciseDuration == 3:
            feature_array[53] = 1  # ExerciseDuration_3
        elif exerciseDuration == 4:
            feature_array[54] = 1  # ExerciseDuration_4

        # 2ly exertion/exertion occupation
        exertionOccupation = data.exertionOccupation
        if exertionOccupation == 1:
            feature_array[55] = 1  # ExertionOccupation_1
        elif exertionOccupation == 2:
            feature_array[56] = 1  # ExertionOccupation_2

        # Duration of sleep per day (hours)
        sleepDuration = data.sleepDuration
        if sleepDuration == 1:
            feature_array[57] = 1  # SleepDuration_1
        elif sleepDuration == 2:
            feature_array[58] = 1  # SleepDuration_2
        elif sleepDuration == 3:
            feature_array[59] = 1  # SleepDuration_3

        # Total QOL
        total_qol = sum([
            data.q1_health, data.q2_pain, data.q3_physical, data.q4_sleep,
            data.q5_feeling, data.q6_concentration, data.q7_self,
            data.q8_bodyImage, data.q9_negativeFeel, data.q10_dailyActivity,
            data.q11_onMedication, data.q12_working, data.q13_relationship,
            data.q14_socialSupport, data.q15_safety, data.q16_home,
            data.q17_financial, data.q18_healthService, data.q19_information,
            data.q20_leisure, data.q21_goodPhysical, data.q22_transportation,
            data.q23_spirituality, data.q24_mobility, data.q25_sex,
            data.q26_level
        ])
        if total_qol < 27:
            feature_array[60] = 1  # _1
        elif 27 <= total_qol <= 95:
            feature_array[61] = 1  # _2
        else:  # total_qol > 95
            feature_array[62] = 1  # _3

        # Serum creatinine
        serumCreatinine = data.creatinine
        if serumCreatinine == 1:
            feature_array[63] = 1  # SerumCreatinine_1
        elif serumCreatinine == 2:
            feature_array[64] = 1  # SerumCreatinine_2
        elif serumCreatinine == 3:
            feature_array[65] = 1  # SerumCreatinine_3

        # Fasting plasma glucose
        fastingPlasmaGlucose = data.plasmaGlucose
        if fastingPlasmaGlucose == 1:
            feature_array[66] = 1  # FastingPlasmaGlucose_1
        elif fastingPlasmaGlucose == 2:
            feature_array[67] = 1  # FastingPlasmaGlucose_2

        # HDL-cholesterol
        hdlCholesterol = data.HDL_cholesterol
        if hdlCholesterol == 1:
            feature_array[68] = 1  # HDLCholesterol_1
        elif hdlCholesterol == 2:
            feature_array[69] = 1  # HDLCholesterol_2
        elif hdlCholesterol == 3:
            feature_array[70] = 1  # HDLCholesterol_3

        # LDL-cholesterol
        ldlCholesterol = data.LDL_cholesterol
        if ldlCholesterol == 1:
            feature_array[71] = 1  # LDLCholesterol_1
        elif ldlCholesterol == 2:
            feature_array[72] = 1  # LDLCholesterol_2
        elif ldlCholesterol == 3:
            feature_array[73] = 1  # LDLCholesterol_3
        elif ldlCholesterol == 4:
            feature_array[74] = 1  # LDLCholesterol_4
        elif ldlCholesterol == 5:
            feature_array[75] = 1  # LDLCholesterol_5

        # Triglycerides
        triglycerides = data.triglyceride
        if triglycerides == 1:
            feature_array[76] = 1  # Triglycerides_1
        elif triglycerides == 2:
            feature_array[77] = 1  # Triglycerides_2
        elif triglycerides == 3:
            feature_array[78] = 1  # Triglycerides_3
        elif triglycerides == 4:
            feature_array[79] = 1  # Triglycerides_4

        # HbA1C
        hba1c = data.HbA1C
        if hba1c == 1:
            feature_array[80] = 1  # HbA1C_1
        elif hba1c == 2:
            feature_array[81] = 1  # HbA1C_2
        elif hba1c == 3:
            feature_array[82] = 1  # HbA1C_3

        # Urine microalbumin
        urineMicroalbumin = data.microalbumin
        if urineMicroalbumin == 1:
            feature_array[83] = 1  # UrineMicroalbumin_1
        elif urineMicroalbumin == 2:
            feature_array[84] = 1  # UrineMicroalbumin_2
        elif urineMicroalbumin == 3:
            feature_array[85] = 1  # UrineMicroalbumin_3

        # แปลงเป็น 2D array
        features = feature_array.reshape(1, -1)

        print("Features:", features)

        # ทำนายผลลัพธ์
        result = model.predict(features)
        print("Prediction Result:", result)
        prediction = model.predict_proba(features)[0]
        risk_score = prediction[1]  # โอกาสเป็นเบาหวาน

        print("Prediction:", prediction)
        
        # กำหนดระดับความเสี่ยง
        if risk_score > 0.5:
            risk_level = "คุณมีความเสี่ยง"
        else:
            risk_level = "คุณไม่มีความเสี่ยง"
        
        return {
            "prediction": float(risk_score),
            "risk_level": risk_level,
            "risk_percentage": float(risk_score * 100),
            "features_used": feature_array.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    try:
        def safe_float(x):
            try:
                value = float(x)
                if np.isinf(value) or np.isnan(value):
                    return 0.0
                return value
            except:
                return 0.0

        # สร้าง safe_params ก่อน
        safe_params = {}
        for k, v in model.get_params().items():
            if isinstance(v, (int, float)):
                safe_params[k] = safe_float(v)
            else:
                safe_params[k] = str(v)

        # จากนั้นค่อยสร้าง feature_importance
        feature_importance = dict(zip(
            ['Height', 'Weight', 'Duration of living in this community (years)',
             'Age', 'Average time spent on social media per day (per hour)', 'Sex',
             'Fatty liver disease', 'Smoking', 'Drinking alcohol',
             'Underlying disease among family', 'Eating Chinese sausages, sausages',
             'Eating fried foods', 'Eating food cooked with coconut milk',
             'Eating Colorful food such as candy, 1 drinks', 'Consumption of tea, coffee (glass)',
             'Eating Nuts or whole grains such as sesame seeds, peanuts, peas, edamame, soybeans.',
             'Exercise (year)', 'Duration of exercise', '2ly exertion/exertion occupation ',
             'Duration of sleep per day (hours)', 'Total QOL', 'Serum creatinine',
             'Fasting plasma glucose', 'HbA1C', 'Plasma Insulin', 'Urine microalbumin'],
            [safe_float(x) for x in model.feature_importances_.tolist()]
        ))

        # เพิ่มโค้ดนี้ชั่วคราวเพื่อตรวจสอบ feature names
        print("Feature count:", len(model.feature_importances_))
        if hasattr(model, 'feature_names_'):
            print("Feature names:", model.feature_names_)
            # เพิ่มหลังจากโหลดโมเดล
            print("Model info:")
            print(model)  # แสดงข้อมูลทั่วไปของโมเดล
            print("\nFeature importances:")
            for i, importance in enumerate(model.feature_importances_):
                print(f"Feature {i}: {importance}")
            print("Feature types:", model.get_booster().feature_types)

        # After loading the model
        print("Model details:")
        print("Number of features:", len(model.feature_importances_))
        print("\nFeature importances and names:")
        importances = model.feature_importances_

        # Try different ways to get feature names
        try:
            # Try to get feature names from DMatrix
            dmatrix = xgb.DMatrix(np.zeros((1, len(importances))))
            feature_names = dmatrix.feature_names
            print("Feature names from DMatrix:", feature_names)
        except:
            pass
        
        try:
            # Try to get feature names from booster
            feature_names = model.get_booster().feature_names
            print("Feature names from booster:", feature_names)
        except:
            pass
        
        # Print raw importances
        print("\nRaw feature importances:")
        for i, imp in enumerate(importances):
            print(f"Feature {i}: {imp}")

        return {
            "model_name": "Diabetes Risk Prediction Model",
            "model_type": f"XGBoost Classifier (version: {xgb.__version__})",
            "features": {
                name: {
                    "description": name,  # Using the feature name as description
                    "importance": safe_float(importance)
                }
                for name, importance in feature_importance.items()
            },
            "model_parameters": safe_params,
            "risk_levels": {
                "มีความเสี่ยง": "risk_score > 0.5",
                "ไม่มีความเสี่ยง": "risk_score <= 0.5"
            },
            "model_file": str(MODEL_PATH),
            "feature_importance_ranking": dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"ไม่สามารถดึงข้อมูลโมเดลได้: {str(e)}"
        )
