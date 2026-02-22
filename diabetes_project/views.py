from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import os
import numpy as np


def load_model():
    try:
        
        data_path = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
        data = pd.read_csv(data_path)
        x = data.drop("Outcome", axis=1)
        y = data["Outcome"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        print("Model trained successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


model = load_model()

def home(request):
    result2 = ""
    
    
    if request.method == 'GET' and all(param in request.GET for param in ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age']):
        # Process the form data
        try:
            val1 = float(request.GET['preg'])
            val2 = float(request.GET['glucose'])
            val3 = float(request.GET['bp'])
            val4 = float(request.GET['skin'])
            val5 = float(request.GET['insulin'])
            val6 = float(request.GET['bmi'])
            val7 = float(request.GET['dpf'])
            val8 = float(request.GET['age'])

            
            features = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
            
           
            if model is not None:
                pred = model.predict(features)
                result2 = "Positive" if pred[0] == 1 else "Negative"
            else:
                result2 = "Error: Model not loaded properly"
            
        except ValueError as e:
            result2 = "Error: Please enter valid numbers in all fields"
        except Exception as e:
            result2 = f"Error in prediction: {str(e)}"
    
    return render(request, "home.html", {"resultx": result2})