from math import log
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pickle
import numpy as np
from pathlib import Path
import xgboost as xgb  # เพิ่ม import xgboost
from ..schemas.userData import UserData

router = APIRouter(
    prefix="/diabetes",
    tags=["diabetes"]
)

# โหลดโมเดล XGBoost
# แก้ไขเส้นทางไฟล์โมเดล
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgb_diabetes_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    raise Exception(f"ไม่สามารถโหลดโมเดลได้: {str(e)}")

@router.post("/predict")
async def predict_risk(data: UserData):
    try:
        print("Received data:", data)

        feature_array = np.zeros(63)  # ปรับขนาดเป็น 63 ตามจำนวน features

        # ตัวแปรต่อเนื่อง
        feature_array[0] = data.height  # Height
        feature_array[1] = data.weight  # Weight
        feature_array[2] = data.livingDuration  # Duration of living

        # Age categories (Age_1, Age_2, Age_3)
        age = data.age
        if age == 1:
            feature_array[62] = 1  # Age_1
        elif age == 2:
            feature_array[3] = 1  # Age_2
        elif age == 3:
            feature_array[4] = 1  # Age_3

        # Social media time (3 categories)
        social_time = data.socialTime
        if social_time == 1:
            feature_array[5] = 1  # _1
        elif social_time == 2:
            feature_array[6] = 1  # _2
        elif social_time == 3:
            feature_array[7] = 1  # _3

        # Sex (2 categories)
        sex = data.sex
        if sex == 1:
            feature_array[8] = 1  # Sex_1
        elif sex == 2:
            feature_array[9] = 1  # Sex_2

        # Fatty liver (2 categories)
        fatty_liver = data.fattyLiver
        if fatty_liver == 1:
            feature_array[10] = 1  # _1
        elif fatty_liver == 2:
            feature_array[11] = 1  # _2

        # Smoking (2 categories)
        smoking = data.smoking
        if smoking == 1:
            feature_array[12] = 1
        elif smoking == 2:
            feature_array[13] = 1

        # Alcohol (2 categories)
        alcohol = data.alcohol
        if alcohol == 1:
            feature_array[14] = 1
        elif alcohol == 2:
            feature_array[15] = 1

        # Disease family (3 categories)
        disease_family = data.diseaseFamily
        if disease_family == 1:
            feature_array[16] = 1
        elif disease_family == 2:
            feature_array[17] = 1
        elif disease_family == 3:
            feature_array[18] = 1

        # Chinese sausages
        if data.eatSausage == 1:
            feature_array[19] = 1
        elif data.eatSausage == 2:
            feature_array[20] = 1

        # Fried foods
        if data.eatFriedFood == 1:
            feature_array[21] = 1
        elif data.eatFriedFood == 2:
            feature_array[22] = 1

        # Coconut milk
        if data.eatCoconutMilk == 1:
            feature_array[23] = 1
        elif data.eatCoconutMilk == 2:
            feature_array[24] = 1

        # Candy and colorful drinks
        if data.eatCandy == 1:
            feature_array[25] = 1
        elif data.eatCandy == 2:
            feature_array[26] = 1

         # Tea/Coffee
        if data.drinkCoffee == 1:
            feature_array[27] = 1
        elif data.drinkCoffee == 2:
            feature_array[28] = 1

        # Nuts and whole grains
        if data.eatNut == 1:
            feature_array[29] = 1
        elif data.eatNut == 2:
            feature_array[30] = 1

        # Exercise (year) (5 categories)
        exercise = data.exercise
        if exercise == 1:
            feature_array[31] = 1  # _1
        elif exercise == 2:
            feature_array[32] = 1  # _2
        elif exercise == 3:
            feature_array[33] = 1  # _3
        elif exercise == 4:
            feature_array[34] = 1  # _4
        elif exercise == 5:
            feature_array[35] = 1  # _5

        # Exercise duration (4 categories)
        exercise_duration = data.exerciseDuration
        if exercise_duration == 1:
            feature_array[36] = 1  # _1
        elif exercise_duration == 2:
            feature_array[37] = 1  # _2
        elif exercise_duration == 3:
            feature_array[38] = 1  # _3
        elif exercise_duration == 4:
            feature_array[39] = 1  # _4

        # Exertion/occupation (2 categories)
        exertion_occupation = data.exertionOccupation
        if exertion_occupation == 1:
            feature_array[40] = 1  # _1
        elif exertion_occupation == 2:
            feature_array[41] = 1  # _2

        # Sleep duration (5 categories)
        sleep_duration = data.sleepDuration
        if sleep_duration == 1:
            feature_array[42] = 1  # _1
        elif sleep_duration == 2:
            feature_array[43] = 1  # _2
        elif sleep_duration == 3:
            feature_array[44] = 1  # _3

        # Total QOL (5 categories)
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
            feature_array[45] = 1  # _1
        elif 27 <= total_qol <= 95:
            feature_array[46] = 1  # _2
        else:  # total_qol > 95
            feature_array[47] = 1  # _3

        # Serum creatinine (3 categories)
        serum_creatinine = data.creatinine
        if serum_creatinine == 1:
            feature_array[48] = 1  # _1
        elif serum_creatinine == 2:
            feature_array[49] = 1  # _2
        else:  # serum_creatinine == 3
            feature_array[50] = 1  # _3

        # Fasting plasma glucose (3 categories)
        fasting_plasma_glucose = data.plasmaGlucose
        if fasting_plasma_glucose == 1:
            feature_array[51] = 1  # _1
        else:  # fasting_plasma_glucose == 2
            feature_array[52] = 1  # _2

        # HbA1C (3 categories)
        hba1c = data.HbA1C
        if hba1c == 1:
            feature_array[53] = 1  # _1
        elif hba1c == 2:
            feature_array[54] = 1  # _2
        else:  # hba1c == 3
            feature_array[55] = 1  # _3

        # Plasma Insulin (3 categories)
        plasma_insulin = data.plasmaInsulin
        if plasma_insulin == 1:
            feature_array[56] = 1  # _1
        elif plasma_insulin == 2:
            feature_array[57] = 1  # _2
        else:  # plasma_insulin == 3
            feature_array[58] = 1  # _3

        # Urine microalbumin (3 categories)
        urine_microalbumin = data.microalbumin
        if urine_microalbumin == 1:
            feature_array[59] = 1  # _1
        elif urine_microalbumin == 2:
            feature_array[60] = 1  # _2
        else:  # urine_microalbumin == 3
            feature_array[61] = 1  # _3

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
