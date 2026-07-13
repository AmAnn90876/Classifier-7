from flask import Flask, render_template, request, jsonify
import pickle
import re
import sklearn

app = Flask(__name__)

# القاموس الفعلي والمطابق لنتائج الـ LabelEncoder في مشروعكِ
category_map = {
    0: "إنارة",
    1: "الإنارة",
    2: "التشوه البصري",
    3: "الحدائق",
    4: "الصيانة",
    5: "الطرق",
    6: "المرور",
    7: "النظافة",
    8: "تشوه بصري",
    9: "تصريف الأمطار",
    10: "حدائق",
    11: "حفريات",
    12: "طرق",
    13: "مبانٍ قابلة للسقوط",
    14: "نظافة"
}

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.linear_model._logistic':
            return super().find_class('sklearn.linear_model', name)
        return super().find_class(module, name)

try:
    with open('model.pkl', 'rb') as f:
        model = CustomUnpickler(f).load()
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = CustomUnpickler(f).load()
    print("✨ تم تحميل النموذج والـ Vectorizer بنجاح!")
except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل الملفات: {e}")

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\d\u0660-\u0669]+', '', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        complaint_text = data.get('complaint', '')
        
        if not complaint_text:
            return jsonify({'error': 'النص فارغ'})
        
        cleaned = clean_text(complaint_text)
        vectorized_text = vectorizer.transform([cleaned])
        
        # الحصول على التنبؤ الرقمي من الموديل
        prediction_numeric = int(model.predict(vectorized_text)[0])
        
        # تحويل الرقم إلى النص العربي المطابق تماماً
        prediction_text = category_map.get(prediction_numeric, f"قسم رقم {prediction_numeric}")
        
        return jsonify({'category': prediction_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 جاري تشغيل السيرفر على الرابط المحلي...")
    app.run(debug=True)