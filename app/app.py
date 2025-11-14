import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/modelo_titanic.joblib")

st.title("🚢 Predicción de Supervivencia - Titanic ML App")
st.write("Ingrese los datos del pasajero para estimar la probabilidad de supervivencia.")

# Pclass
pclass = st.selectbox("Clase del pasajero (1 = Primera, 2 = Segunda, 3 = Tercera)", [1, 2, 3])

# Age
age = st.number_input("Edad", min_value=0, max_value=100, value=30)

# SibSp
sibsp = st.number_input("Número de hermanos/esposos a bordo (SibSp)", min_value=0, max_value=10, value=0)

# Parch
parch = st.number_input("Número de padres/hijos a bordo (Parch)", min_value=0, max_value=10, value=0)

# Fare
fare = st.number_input("Tarifa pagada (Fare)", min_value=0.0, value=7.25, format="%.3f")

# Genero
genero = st.selectbox("Género", ["male", "female"])

if genero == "male":
    male = 1
    female = 0
else:
    male = 0
    female = 1

# Puerto de embarque
embarked = st.selectbox("Puerto de embarque", ["C", "Q", "S"])

C = 1 if embarked == "C" else 0
Q = 1 if embarked == "Q" else 0
S = 1 if embarked == "S" else 0

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "female": [female],
    "male": [male],
    "C": [C],
    "Q": [Q],
    "S": [S]
})


if st.button("🔍 Predecir"):
    
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Probabilidad de sobrevivir

    st.subheader("📊 Resultado")

    if pred == 1:
        st.success(f"El pasajero **SOBREVIVE** con una probabilidad del {proba*100:.2f}%")
    else:
        st.error(f"El pasajero **NO SOBREVIVE**. Probabilidad de supervivencia: {proba*100:.2f}%")

    st.write("Datos utilizados para la predicción:")
    st.dataframe(input_data)
