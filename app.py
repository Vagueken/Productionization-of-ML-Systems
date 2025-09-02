from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('flight_price_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Define training features exactly as seen during fit
training_features = [
    'time', 'distance', 'day_of_week', 'month',
    'from_Aracaju (SE)', 'from_Brasilia (DF)', 'from_Campo Grande (MS)',
    'from_Florianopolis (SC)', 'from_Natal (RN)', 'from_Recife (PE)',
    'from_Rio de Janeiro (RJ)', 'from_Salvador (BH)', 'from_Sao Paulo (SP)',
    'to_Aracaju (SE)', 'to_Brasilia (DF)', 'to_Campo Grande (MS)',
    'to_Florianopolis (SC)', 'to_Natal (RN)', 'to_Recife (PE)',
    'to_Rio de Janeiro (RJ)', 'to_Salvador (BH)', 'to_Sao Paulo (SP)',
    'flightType_economic', 'flightType_firstClass', 'flightType_premium',
    'agency_CloudFy', 'agency_FlyingDrops', 'agency_Rainbow'
]

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            data = request.json
        elif request.method == 'GET':
            data = {
                'from': request.args.get('from', 'Rio de Janeiro (RJ)'),
                'to': request.args.get('to', 'Sao Paulo (SP)'),
                'flightType': request.args.get('flightType', 'economic'),
                'agency': request.args.get('agency', 'CloudFy'),
                'time': float(request.args.get('time', 6)),
                'distance': float(request.args.get('distance', 5000)),
                'day_of_week': float(request.args.get('day_of_week', 2)),
                'month': float(request.args.get('month', 7))
            }
        # Create initial DataFrame with input data
        df = pd.DataFrame([data])
        # Encode categorical columns
        cat_cols = ['from', 'to', 'flightType', 'agency']
        encoded_cols = pd.DataFrame(encoder.transform(df[cat_cols]))
        # Ensure column names match training (handle case sensitivity)
        encoded_cols.columns = [col.lower() for col in encoder.get_feature_names_out(cat_cols)]
        col_mapping = {col.lower(): col for col in training_features if '_' in col}
        encoded_cols_renamed = encoded_cols.rename(columns=col_mapping, errors='ignore')
        # Combine with numerical columns
        num_cols = ['time', 'distance', 'day_of_week', 'month']
        df_processed = pd.concat([df[num_cols], encoded_cols_renamed], axis=1)
        # Align with training features, filling missing columns with 0
        df_aligned = pd.DataFrame(0, index=df_processed.index, columns=training_features)
        for col in df_processed.columns:
            if col in training_features:
                df_aligned[col] = df_processed[col]
        # Scale only the numerical part (scaler expects all num_cols including price)
        num_cols_full = ['price', 'time', 'distance', 'day_of_week', 'month']
        df_scaled_input = pd.DataFrame(0, index=df_aligned.index, columns=num_cols_full)
        df_scaled_input[num_cols] = df_aligned[num_cols]  # Fill numerical features
        df_scaled_num = scaler.transform(df_scaled_input)
        # Combine scaled numerical features with categorical features
        df_scaled_full = np.hstack((df_scaled_num[:, 1:], df_aligned.drop(columns=num_cols).values))  # Exclude price, add categoricals
        # Predict
        pred_scaled = model.predict(df_scaled_full)
        # Unscale the prediction
        pred_unscaled = scaler.inverse_transform(
            np.concatenate([pred_scaled.reshape(-1, 1), np.zeros((len(pred_scaled), len(num_cols_full)-1))], axis=1)
        )[:, 0]
        return jsonify({'price': pred_unscaled[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)