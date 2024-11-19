import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleme
file_path = 'datasets/german_credit_data.csv'
data = pd.read_csv(file_path)
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

data['Saving accounts'] = data['Saving accounts'].fillna('unknown')
data['Checking account'] = data['Checking account'].fillna('unknown')

# Sütunları ayırma
numerical_columns = [col for col in data.columns if data[col].nunique() > 10]
categorical_columns = [col for col in data.columns if col not in numerical_columns]

# Yaş kategorisi ekleme
interval = (18, 25, 35, 60, 120)
cats = ['Student', 'Young', 'Adult', 'Senior']
data['Age_cat'] = pd.cut(data['Age'], interval, labels=cats)

# Risk ve diğer kategorik sütunları encode etme
label_encoders = {}
for col in ['Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age_cat', 'Sex', 'Risk']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Model için X ve y ayrımı
X = data.drop(columns=['Risk', 'Sex'])
y = data['Risk']

# Veriyi train-test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit tasarımı
st.set_page_config(page_title="Kredi Risk Tahmini", layout="wide")
st.title("Kredi Risk Tahmini")
st.markdown(
    "Bu uygulama, kullanıcı bilgilerinden kredi riskini tahmin etmek için geliştirilmiştir. "
    "Lütfen sol taraftaki formu doldurun ve tahmin sonucunu görmek için butona tıklayın."
)

def get_user_input():
    st.sidebar.subheader("Kullanıcı Bilgilerini Girin")
    age = st.sidebar.number_input("Yaş", min_value=18, max_value=100, value=30)
    job = st.sidebar.number_input("Meslek Seviyesi (0-3)", min_value=0, max_value=3, value=0)
    housing = st.sidebar.selectbox("Konut Durumu", label_encoders['Housing'].classes_)
    saving_accounts = st.sidebar.selectbox("Birikim Hesabı Durumu", label_encoders['Saving accounts'].classes_)
    checking_account = st.sidebar.selectbox("Çek Hesabı Durumu", label_encoders['Checking account'].classes_)
    credit_amount = st.sidebar.number_input("Kredi Miktarı", min_value=0, value=1000)
    duration = st.sidebar.number_input("Kredi Süresi (Ay)", min_value=1, value=12)
    purpose = st.sidebar.selectbox("Kredi Amacı", label_encoders['Purpose'].classes_)

    user_input = {
        'Age': age,
        'Job': job,
        'Housing': label_encoders['Housing'].transform([housing])[0],
        'Saving accounts': label_encoders['Saving accounts'].transform([saving_accounts])[0],
        'Checking account': label_encoders['Checking account'].transform([checking_account])[0],
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': label_encoders['Purpose'].transform([purpose])[0],
        'Age_cat': label_encoders['Age_cat'].transform(['Young'])[0]
    }
    return user_input

# Kullanıcı girdileri
user_input = get_user_input()
if st.sidebar.button("Tahmin Et"):
    user_df = pd.DataFrame([user_input])
    user_df = user_df.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(user_df)[0]
    result = "✔ Risksiz (1)" if prediction == 1 else "❌ Riskli (0)"

    st.success(f"Tahmini Sonuç: {result}")


