import pandas as pd
from sklearn.ensemble import IsolationForest
from flask import Flask, request, jsonify

# Load dữ liệu giao dịch (ví dụ)
data = pd.read_csv('diabetes.csv')

# Khởi tạo mô hình Isolation Forest
model = IsolationForest(contamination=0.01)  # 1% của dữ liệu được coi là gian lận

# Huấn luyện mô hình trên dữ liệu hiện tại
model.fit(data)

# Khởi tạo Flask app
app = Flask(__name__)

# API endpoint cho việc dự đoán
@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    # Nhận dữ liệu từ yêu cầu POST
    transaction_data = request.json

    # Dự đoán bằng mô hình đã được huấn luyện
    predictions = model.predict(transaction_data)

    # Chuẩn bị kết quả dự đoán
    result = {'fraud_prediction': predictions.tolist()}

    return jsonify(result)

# Chạy Flask app
if __name__ == '__main__':
    app.run(debug=True)