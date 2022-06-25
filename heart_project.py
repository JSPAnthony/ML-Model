# Import of needed libraries
import pickle
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app object
app = FastAPI()

# Load trained Pipeline
# model = load_model('heart_disease.pkl')
model = pickle.load(open('heart_disease.pkl', 'rb'))

# Define predict function

@app.post('/predict')
def predict(sex, chestpaintype, restingecg, exerciseangina, st_slope, age, restingbp, cholesterol,
            fastingbs, maxhr, oldpeak):
    data = pd.DataFrame([[sex, chestpaintype, restingecg, exerciseangina, st_slope, age, restingbp,
                          cholesterol, fastingbs, maxhr, oldpeak]])
    data.columns = ['sex', 'chestpaintype', 'restingecg', 'exerciseangina', 'st_slope', 'age', 'restingbp',
                    'cholesterol', 'fastingbs', 'maxhr', 'oldpeak']

    predictions = predict_model(model, data=data)
    return {'prediction': int(predictions['Label'][0])}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
