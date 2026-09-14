# BlackBox — AI Predictive Maintenance

## هيكل المشروع

- data/raw          -> البيانات الخام زي ما نزلتها (NASA Bearing Dataset)
- data/processed     -> بيانات بعد التنظيف/المعالجة، جاهزة للموديل
- ml                -> notebooks وسكريبتات تدريب الموديل (Isolation Forest)
- dashboard         -> كود الداشبورد (Streamlit)
- hardware-sim      -> سكريبت محاكاة بيانات الحساسات
- docs              -> التوثيق والعرض التقديمي

## خطوات التشغيل

1. python3 -m venv venv
2. source venv/bin/activate   (Windows: venv\Scripts\activate)
3. pip install -r requirements.txt
4. نزّلي الداتا في data/raw (راجعي docs للخطوات)
