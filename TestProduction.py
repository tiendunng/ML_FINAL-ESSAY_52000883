from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, request, jsonify

# Khởi tạo mô hình dịch
model_name = "Helsinki-NLP/opus-mt-en-vi"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Khởi tạo Flask app
app = Flask(__name__)

# API endpoint cho việc dự đoán
@app.route('/translate', methods=['POST'])
def translate_text():
    # Nhận dữ liệu từ yêu cầu POST
    input_text = request.json.get('input_text', '')
    
    # Tokenize và dự đoán bằng mô hình đã được huấn luyện
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    translation_ids = model.generate(input_ids, max_length=50, num_beams=5)
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)

    # Chuẩn bị kết quả dịch
    result = {'translated_text': translated_text}

    return jsonify(result)

# Chạy Flask app
if __name__ == '__main__':
    app.run(debug=True)
