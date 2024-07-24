from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        'annual_income', 'monthly_inhand_salary', 'num_bank_accounts',
       'num_credit_card', 'interest_rate', 'num_of_loan',
       'delay_from_due_date', 'num_of_delayed_payment', 'changed_credit_limit',
       'num_credit_inquiries','credit_mix', 'outstanding_debt', 'credit_utilization_ratio',
       'credit_history_age', 'total_emi_per_month', 'amount_invested', 'monthly_balance'
    ]
    
    input_data = [request.form[feature] for feature in features]
    input_data = [0.0 if value == '' else float(value) for value in input_data]
    input_data = np.array(input_data, dtype=float).reshape(1, -1)
    
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        result = "Good"
    elif prediction[0]==1:
        result = "Poor"
    else:
        result="Standard"
    
    return render_template('result.html', prediction_text=f"Credit Score: {result}")

if __name__ == '__main__':
    app.run(port=8000, debug=True)