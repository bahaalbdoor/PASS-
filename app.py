import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# 🎨 إعدادات الألوان المطلوبة
PRIMARY_COLOR = "#20A8E0"
PRIMARY_DARK = "#1a8ab8"
BG_COLOR = "#f4f7f6"
CARD_BG = "#ffffff"
TEXT_COLOR = "#2c3e50"
OUTPUT_BG = "#f8f9fa"

# 🖥️ إعداد الصفحة وتصميم CSS مخصص
st.set_page_config(page_title="PASS - نظام التنبؤ بالنجاح", page_icon="🎓", layout="wide")
st.markdown(f"""
<style>
/* الخلفية العامة */
.main {{
    background-color: {BG_COLOR};
}}

/* الحاوية الرئيسية */
.main .block-container {{
    max-width: 1300px;
    margin: 0 auto;
    padding: 2.5rem 2rem;
    background-color: {CARD_BG};
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    text-align: center;
}}

/* العناوين والنصوص */
h1, h2, h3, h4, h5, p, label, span, li {{
    color: {TEXT_COLOR};
    font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
}}
h1 {{ font-size: 2.6rem; font-weight: 800; margin-bottom: 0.5rem; text-align: center; }}
h2 {{ font-size: 1.8rem; font-weight: 700; margin-top: 1.5rem; text-align: center; }}

/* الشعار */
.logo-container {{
    text-align: center;
    margin-bottom: 0px;
}}
.logo-container img {{
    max-width: 200px;
    height: auto;
    border-radius: 12px;
}}

/* التبويبات - عكس الاتجاه */
.stTabs [data-baseweb="tab-list"] {{
    justify-content: center;
    background-color: {BG_COLOR};
    padding: 8px;
    border-radius: 12px;
    gap: 8px;
    flex-direction: row-reverse !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {CARD_BG};
    color: {TEXT_COLOR};
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 700;
    border: 2px solid transparent;
    transition: all 0.2s ease;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border-color: {PRIMARY_DARK};
    box-shadow: 0 4px 12px rgba(32, 168, 224, 0.3);
}}

/* حاوية حقول الإدخال */
.input-container {{
    max-width: 650px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 15px;
}}

.stSlider, .stSelectbox, .stRadio {{
    width: 100%;
    background-color: {BG_COLOR};
    padding: 12px 15px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}}

.stRadio > div {{
    flex-direction: row !important;
    justify-content: center;
    gap: 25px;
}}

/* الأزرار */
.stButton {{
    width: 100%;
    display: flex;
    justify-content: center;
    margin-top: 25px;
}}
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY_COLOR}, {PRIMARY_DARK});
    color: white;
    border: none;
    padding: 16px 45px;
    font-size: 1.15rem;
    font-weight: 800;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px rgba(32, 168, 224, 0.4);
    max-width: 500px;
    width: 100%;
}}
.stButton > button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(32, 168, 224, 0.6);
}}

/* النتائج والمخرجات */
div[data-testid="stSuccess"] {{
    background-color: {OUTPUT_BG};
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 2px solid {PRIMARY_COLOR};
}}
div[data-testid="stMetric"] {{
    background-color: {OUTPUT_BG};
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    border: 2px solid {PRIMARY_COLOR};
}}

/* الجداول */
div[data-testid="stDataFrame"] {{
    background-color: {CARD_BG};
    border-radius: 15px;
    padding: 15px;
}}
div[data-testid="stDataFrame"] th {{
    background-color: {PRIMARY_COLOR};
    color: white;
    text-align: center !important;
    padding: 14px;
}}
div[data-testid="stDataFrame"] td {{
    text-align: center !important;
    padding: 12px;
    background-color: {BG_COLOR};
}}

/* شريط التقدم */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {PRIMARY_COLOR}, {PRIMARY_DARK});
    border-radius: 10px;
}}

/* التذييل */
footer, .stCaption {{
    text-align: center !important;
    color: {TEXT_COLOR};
    opacity: 0.75;
    margin-top: 30px;
}}

/* إخفاء عناصر Streamlit */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* ✅ CSS قوي جداً لمحتوى تبويب الشرح - مركزي تماماً */
#tab-explain-center {{
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center !important;
}}
#tab-explain-center * {{
    text-align: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
}}
#tab-explain-center .stColumns {{
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}}
#tab-explain-center [data-testid="stMetric"] {{
    margin: 0 10px !important;
}}
#tab-explain-center ul, #tab-explain-center li {{
    text-align: right !important;
    display: inline-block !important;
    max-width: 800px !important;
}}
.st-emotion-cache-3pwa5w{{
text-align: center !important;
}}
.st-emotion-cache-yn44r9 > ul, .st-emotion-cache-yn44r9 > ol
{{
text-align: right !important;
  direction: rtl !important;
  justify-content: center !important;
  display: inline flow-root list-item !important;
}}
.st-emotion-cache-wfksaw
{{

  direction: rtl;
}}
</style>
""", unsafe_allow_html=True)

st.title(" PASS - نظام التنبؤ بالنجاح الأكاديمي")
st.markdown(
    f'<p style="font-size: 1.1rem; color: {TEXT_COLOR}; margin-bottom: 2rem; text-align: center;">نظام ذكي للتنبؤ بنتيجة الطالب بناءً على بيانات UCI الرسمية (Math Course)</p>',
    unsafe_allow_html=True)

# عرض الشعار
try:
    if os.path.exists("logo.png"):
        import base64

        with open("logo.png", "rb") as f:
            logo_data = f.read()
            logo_base64 = base64.b64encode(logo_data).decode()
        st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" alt="شعار المدرسة">
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    pass


# تحميل البيانات
@st.cache_data
def load_and_prepare_data():
    file_path = "student-mat.csv"
    if not os.path.exists(file_path):
        st.error("⚠️ ملف `student-mat.csv` غير موجود في مجلد التطبيق!")
        st.info("💡 يرجى استخراج الملف من `student.zip` ووضعه بجانب `app.py` ثم إعادة تحميل الصفحة.")
        st.stop()

    df = pd.read_csv(file_path, sep=";")
    target_cols = ["studytime", "absences", "failures", "higher", "internet", "famrel", "G3"]
    df = df[target_cols].copy()
    df = df.drop_duplicates().dropna()
    df["Result"] = (df["G3"] >= 10).map({True: "ناجح ✅", False: "راسب ❌"})
    df = df.sort_values(by="G3", ascending=False).reset_index(drop=True)
    return df


df = load_and_prepare_data()


# تدريب النموذج
@st.cache_resource
def train_model(data):
    features = ["studytime", "absences", "failures", "higher", "internet", "famrel"]
    X = pd.get_dummies(data[features], drop_first=True)
    y = data["Result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features


model, accuracy, feature_cols = train_model(df)

# 🔀 تبويبات الواجهة
tab_pred, tab_data, tab_explain = st.tabs(["🔮 التنبؤ الفوري", "📊 عرض البيانات", "💡 شرح"])

# ================= TAB 1: التنبؤ =================
with tab_pred:
    st.header("🔮 أدخل بيانات طالب جديد")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        studytime = st.radio(
            "⏱️ وقت الدراسة الأسبوعي",
            options=[1, 2, 3, 4],
            index=1,
            help="1: <2 ساعة | 2: 2-5 ساعات | 3: 5-10 ساعات | 4: >10 ساعات",
            horizontal=True
        )

        absence_options = {
            "0 أيام": 0,
            "1-5 أيام": 3,
            "6-15 يوم": 10,
            "16-30 يوم": 23,
            "31+ يوم": 40
        }
        absence_labels = list(absence_options.keys())
        absence_selected = st.radio(
            "📅 عدد الغيابات",
            options=absence_labels,
            index=1,
            horizontal=True
        )
        absences = absence_options[absence_selected]

        failures = st.radio(
            "❌ رسوبات سابقة",
            options=[0, 1, 2, 3, 4],
            index=0,
            horizontal=True
        )

        higher = st.radio("🎓 يطمح للتعليم العالي؟", ["نعم", "لا"], index=0, horizontal=True)
        internet = st.radio("💻 إنترنت في المنزل؟", ["نعم", "لا"], index=0, horizontal=True)

        famrel = st.radio(
            "👧 جودة العلاقة الأسرية",
            options=[1, 2, 3, 4, 5],
            index=3,
            help="1: ضعيفة جداً | 5: ممتازة",
            horizontal=True
        )

        if st.button("🚀 احسب النتيجة المتوقعة", use_container_width=True):
            higher_val = "yes" if higher == "نعم" else "no"
            internet_val = "yes" if internet == "نعم" else "no"

            inp = pd.DataFrame([{
                "studytime": studytime, "absences": absences, "failures": failures,
                "higher": higher_val, "internet": internet_val, "famrel": famrel
            }])
            inp_encoded = pd.get_dummies(inp, drop_first=True)
            for col in model.feature_names_in_:
                if col not in inp_encoded.columns:
                    inp_encoded[col] = 0
            inp_encoded = inp_encoded[model.feature_names_in_]

            pred = model.predict(inp_encoded)[0]
            prob = model.predict_proba(inp_encoded)[0].max()

            st.success(f"🎯 التوقع: **{pred}**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📈 ثقة النموذج", f"{prob:.1%}")
            with col_b:
                st.progress(prob)

# ================= TAB 2: عرض البيانات =================
with tab_data:
    st.header("📊 عينة من البيانات الحقيقية")
    display_df = df.head(15).copy()
    display_df.columns = ["⏱️ دراسة", "📅 غياب", "❌ رسوب", "🎓 طموح", "💻 إنترنت", "👩‍ أسرة", "📊 درجة", "🎯 نتيجة"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 تحميل الملف النهائي (CSV)", data=csv, file_name="PASS_dataset_clean.csv", mime="text/csv")

# ================= TAB 3: الشرح - محتوى متمركز تماماً =================
with tab_explain:
    # استخدام div مع ID محدد لتطبيق CSS قوي
    st.markdown('<div id="tab-explain-center">', unsafe_allow_html=True)

    st.header("💡 كيف يعمل النظام؟")

    # أعمدة المقاييس - 3 أعمدة متساوية ومتمركزة
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 دقة النموذج", f"{accuracy * 100:.1f}%")
    with col2:
        st.metric("📚 عدد الطلاب", f"{len(df)}")
    with col3:
        st.metric("🔍 العوامل المؤثرة", f"{len(feature_cols)}")

    st.divider()

    # محتوى الشرح - نص متمركز
    st.markdown("""
    ### 🧠 المنطق العلمي وراء النظام:

    - **المصدر**: بيانات أكاديمية حقيقية من جامعة البرتغال (مستودع UCI)
    - **الهدف**: التنبؤ بـ (ناجح ✅ / راسب ❌) بناءً على الدرجة النهائية
    - **الخوارزمية**: `Logistic Regression` لسهولة تفسيرها
    - **الدقة**: النموذج دقيق وموثوق في توقعاته
    """)

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption(f"PASS v1.0 | Predictive Academic Success System | Coded By Baha' Al-Bdour | {pd.Timestamp.now().year}")
