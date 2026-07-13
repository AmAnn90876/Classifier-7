import os
import re
import pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from supabase import create_client, Client

app = Flask(__name__)

# إعداد الاتصال بقاعدة بيانات Supabase السحابية
SUPABASE_URL = "https://asjhgyhvngmbbevrzfjm.supabase.co"
SUPABASE_KEY = "ضعي_هنا_مفتاح_sb_publishable_الذي_نسختيه"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# القاموس المطابق لنتائج الموديل
category_map = {
    0: "إنارة", 1: "الإنارة", 2: "التشوه البصري", 3: "الحدائق", 4: "الصيانة",
    5: "الطرق", 6: "المرور", 7: "النظافة", 8: "تشوه بصري", 9: "تصريف الأمطار",
    10: "حدائق", 11: "حفريات", 12: "طرق", 13: "مبانٍ قابلة للسقوط", 14: "نظافة"
}

# تحميل الموديل والـ Vectorizer بإعداداتك الخاصة (Unpickler المخصص)
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.linear_model._logistic':
            return super().find_class('sklearn.linear_model', name)
        return super().find_class(module, name)

with open('model.pkl', 'rb') as f:
    model = CustomUnpickler(f).load()
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = CustomUnpickler(f).load()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\d\u0660-\u0669]+', '', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        complaint_text = data.get('complaint', '')
        if not complaint_text: return jsonify({'error': 'النص فارغ'})
        
        cleaned = clean_text(complaint_text)
        vectorized = vectorizer.transform([cleaned])
        
        pred_numeric = int(model.predict(vectorized)[0])
        category = category_map.get(pred_numeric, "غير معروف")
        confidence = float(max(model.predict_proba(vectorized)[0]))
        
        # حفظ البلاغ مباشرة في قاعدة بيانات Supabase السحابية
        report_data = {
            "name": data.get('name', 'reham'),  # يأخذ الاسم المرسل أو يضع اسم افتراضي
            "details": complaint_text,
            "category": category
        }
        supabase.table('reports').insert(report_data).execute()
        
        return jsonify({'category': category, 'confidence': round(confidence, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/complaints')
def get_complaints():
    try:
        # جلب البلاغات من Supabase وترتيبها من الأحدث للأقدم
        response = supabase.table('reports').select('*').order('created_at', desc=True).execute()
        complaints = response.data
        
        data = []
        for c in complaints:
            # تحويل صيغة الوقت القادمة من Supabase لتطابق التصميم الحالي لديكِ
            try:
                dt = datetime.strptime(c['created_at'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                formatted_time = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M')

            data.append({
                'timestamp': formatted_time,
                'text': c['details'],
                'category': c['category'],
                'confidence': "100.0%"  # قيمة افتراضية متوافقة مع شكل جدولك الحالي
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
