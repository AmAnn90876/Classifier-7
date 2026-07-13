from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import re
import os

app = Flask(__name__)

# إعداد قاعدة البيانات (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///complaints.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# القاموس المطابق لنتائج الموديل
category_map = {
    0: "إنارة", 1: "الإنارة", 2: "التشوه البصري", 3: "الحدائق", 4: "الصيانة",
    5: "الطرق", 6: "المرور", 7: "النظافة", 8: "تشوه بصري", 9: "تصريف الأمطار",
    10: "حدائق", 11: "حفريات", 12: "طرق", 13: "مبانٍ قابلة للسقوط", 14: "نظافة"
}

# تعريف جدول الشكاوى في قاعدة البيانات
class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# تحميل الموديل والـ Vectorizer (بإعداداتك الخاصة)
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

# إنشاء الجداول عند بدء التشغيل
with app.app_context():
    db.create_all()

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
        
        # حفظ في قاعدة البيانات
        new_complaint = Complaint(text=complaint_text, category=category, confidence=confidence)
        db.session.add(new_complaint)
        db.session.commit()
        
        return jsonify({'category': category, 'confidence': round(confidence, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/complaints')
def get_complaints():
    complaints = Complaint.query.order_by(Complaint.timestamp.desc()).all()
    data = [{
        'timestamp': c.timestamp.strftime('%Y-%m-%d %H:%M'),
        'text': c.text,
        'category': c.category,
        'confidence': f"{c.confidence * 100:.1f}%"
    } for c in complaints]
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
