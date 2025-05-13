from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load models and encoders
classifier = joblib.load('multi_output_classifier_ok.pkl')
regressor = joblib.load('regression_model_ok.pkl')
label_encoders = joblib.load('label_encoders_ok.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    stress = int(request.form['Stress_Level'])
    sleep = int(request.form['Sleep_Level'])
    d1 = request.form['Disease_1']
    d2 = request.form['Disease_2'] or 'NaN'
    d3 = request.form['Disease_3'] or 'NaN'

    input_data = {
        'Stress_Level': stress,
        'Sleep_Level': sleep,
        'Disease_1': d1,
        'Disease_2': d2,
        'Disease_3': d3
    }

    df = pd.DataFrame([input_data])

    for col in ['Disease_1', 'Disease_2', 'Disease_3']:
        df[col] = label_encoders[col].transform(df[col])

    category_pred = classifier.predict(df)[0]
    improvement_pred = regressor.predict(df)[0]

    output_labels = ['Nutrient_1', 'Nutrient_2', 'Nutrient_3',
                     'Ingredient_1', 'Ingredient_2', 'Ingredient_3']
    decoded = {
        label: label_encoders[label].inverse_transform([category_pred[i]])[0]
        for i, label in enumerate(output_labels)
    }
    decoded['Improvement_%'] = f"{improvement_pred:.2f}%"

    return render_template('index.html', prediction=decoded, inputs=input_data)
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
