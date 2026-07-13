import os
import re
import pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from supabase import create_client, Client

app = Flask(__name__)

# إعداد الاتصال بقاعدة بيانات Supabase السحابية
SUPABASE_URL = "https://asjhgyhvngmbbevrzfjm.supabase.co"
SUPABASE_KEY = "sb_publishable_B6VEA-7t67Fk0NMisAQg7A_VfpsBge1"

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
        
        # حفظ البلاغ مباشرة في قاعدة بيانات Supabase مع رقم الهاتف والاسم
        report_data = {
            "name": data.get('name', 'غير معرف'),
            "phone": data.get('phone', ''),
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
        user_phone = request.args.get('phone')
        query = supabase.table('reports').select('*')
        
        if user_phone:
            query = query.eq('phone', user_phone)
            
        response = query.order('created_at', desc=True).execute()
        complaints = response.data
        
        data = []
        for c in complaints:
            try:
                dt = datetime.strptime(c['created_at'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                formatted_time = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M')

            data.append({
                'timestamp': formatted_time,
                'text': c['details'],
                'category': c['category'],
                'confidence': "100.0%"
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

# مسار الـ API الجديد لحساب إحصائيات لوحة التحكم الشاملة
@app.route('/api/dashboard-stats')
def get_dashboard_stats():
    try:
        # جلب كل البلاغات لتحليلها
        response = supabase.table('reports').select('*').order('created_at', desc=True).execute()
        reports = response.data
        
        total_count = len(reports)
        
        # حساب تكرار كل قسم للرسم البياني
        category_counts = {}
        for r in reports:
            cat = r.get('category', 'غير معروف')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
        # تنسيق البلاغات الأخيرة لعرضها في جدول الـ Dashboard الشامل
        formatted_reports = []
        for r in reports[:10]: # سنعرض آخر 10 بلاغات في الصفحة الرئيسية للإيجاز
            try:
                dt = datetime.strptime(r['created_at'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                formatted_time = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
                
            formatted_reports.append({
                'name': r.get('name', 'غير معرف'),
                'phone': r.get('phone', 'غير متوفر'),
                'text': r.get('details', ''),
                'category': r.get('category', 'غير معروف'),
                'timestamp': formatted_time
            })
            
        return jsonify({
            'total_reports': total_count,
            'category_distribution': category_counts,
            'recent_reports': formatted_reports
        })
    except Exception as e:
        return jsonify({'error': str(e)})
from flask import Flask, render_template, jsonify, request


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
