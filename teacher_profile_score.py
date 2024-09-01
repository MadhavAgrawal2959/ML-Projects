import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Load the Data
data = pd.read_excel('./teacher_profile_scores_12_months.xlsx')  # Replace with the actual path to your dataset

# Step 3: Split the Data
X = data.drop(columns=['Profile Score', 'Teacher Name'])  # Features
y = data['Profile Score']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# FastAPI app initialization
app = FastAPI()

# Pydantic model for input data
class TeacherProfileData(BaseModel):
    Month : int
    Attendance : int
    Feedback_from_Students : float
    Questions_answered :int
    Research_Papers : int
    Avg_class_duration : int
    Avg_stud_Attedance_per_class : int
    Resource_sharing:float
    Quiz_frequency : int
    After_quiz_feedback :float


# Endpoint to get the prediction
@app.post("/predict/")
async def predict_score(profile_data: TeacherProfileData):
    data_dict = profile_data.dict().values()
    # Convert data to the format expected by the model
    data_df = pd.DataFrame([data_dict])
    print(data_df)
    # Predict using the trained model
    prediction = rf_model.predict(data_df)
    return {"predicted_profile_score": prediction[0]}

