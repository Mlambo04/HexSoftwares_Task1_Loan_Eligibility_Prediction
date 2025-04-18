from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Loan Prediction API',
          description='ML Model for Loan Eligibility Prediction')

ns = api.namespace('predict', description='Prediction operations')

model = joblib.load('best_loan_model.joblib')
feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                'Loan_Amount_Term', 'Credit_History', 'Property_Area_Rural',
                'Property_Area_Semiurban', 'Property_Area_Urban']

loan_model = api.model('LoanParams', {
    'Gender': fields.Float(required=True, description='Male (1) or Female (0)'),
    'Married': fields.Float(required=True, description='Married (1) or Not (0)'),
    'Dependents': fields.Float(required=True, description='Number of dependents'),
    'Education': fields.Float(required=True, description='Graduate (1) or Not (0)'),
    'Self_Employed': fields.Float(required=True, description='Yes (1) or No (0)'),
    'ApplicantIncome': fields.Float(required=True, description='Monthly income'),
    'CoapplicantIncome': fields.Float(required=True, description='Co-applicant income'),
    'LoanAmount': fields.Float(required=True, description='Loan amount in thousands'),
    'Loan_Amount_Term': fields.Float(required=True, description='Term in months'),
    'Credit_History': fields.Float(required=True, description='Credit history meets guidelines'),
    'Property_Area': fields.String(required=True, description='Rural/Semiurban/Urban')
})

@ns.route('/')
class LoanPrediction(Resource):
    @api.expect(loan_model)
    def post(self):
        """Make loan prediction"""
        try:
            data = request.json
            property_area = data.pop('Property_Area')
            
            # Convert property area to one-hot encoding
            areas = ['Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
            encoded = {k: 1 if property_area.lower() in k.lower() else 0 for k in areas}
            
            features = pd.DataFrame([{**data, **encoded}])
            features = features[feature_names].values.reshape(1, -1)
            
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            return {
                'status': 'Approved' if prediction else 'Rejected',
                'probability': float(probability),
                'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
            }
        except Exception as e:
            return {'error': str(e)}, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)