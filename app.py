import streamlit as st
import pandas as pd
import joblib

st.title("Probabilidad de default de tarjeta de credito en Taiwan")

# Input fields
Monto = st.slider("Monto", min_value=10000, max_value=740000, value=10000)
Genero = st.selectbox("Genero (1= hombre, 2=mujer)", [1, 2])
Educacion = st.selectbox("Educacion (1=posgrado, 2=universidad, 3=secundario, 4=otro)", [1, 2, 3, 4])
Estado_Civil = st.selectbox("Estado civil (1=casado, 2=soltero, 3=otro)", [1, 2, 3])
Edad = st.slider("Edad", min_value=21, max_value=100, value=21)
H1 = st.selectbox("Demora en el pago el mes 6", [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
H2 = st.selectbox("Demora en el pago el mes 5", [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
H3 = st.selectbox("Demora en el pago el mes 4", [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
H4 = st.selectbox("Demora en el pago el mes 3", [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
H5 = st.selectbox("Demora en el pago el mes 2", [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
H6 = st.selectbox("Demora en el pago el mes 1", [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
MC1 = st.slider("Saldo en la cuenta el mes 6", min_value=-340000, max_value=600000, value=0)
MC2 = st.slider("Saldo en la cuenta el mes 5", min_value=-340000, max_value=600000, value=0)
MC3 = st.slider("Saldo en la cuenta el mes 4", min_value=-340000, max_value=600000, value=0)
MC4 = st.slider("Saldo en la cuenta el mes 3", min_value=-340000, max_value=600000, value=0)
MC5 = st.slider("Saldo en la cuenta el mes 2", min_value=-340000, max_value=600000, value=0)
MC6 = st.slider("Saldo en la cuenta el mes 1", min_value=-340000, max_value=600000, value=0)
MP1 = st.slider("Importe pagado el mes 6", min_value=0, max_value=420000, value=0)
MP2 = st.slider("Importe pagado el mes 5", min_value=0, max_value=420000, value=0)
MP3 = st.slider("Importe pagado el mes 4", min_value=0, max_value=420000, value=0)
MP4 = st.slider("Importe pagado el mes 3", min_value=0, max_value=420000, value=0)
MP5 = st.slider("Importe pagado el mes 2", min_value=0, max_value=420000, value=0)
MP6 = st.slider("Importe pagado el mes 1", min_value=0, max_value=420000, value=0)

model = joblib.load(r'rf_taiwan_grid.joblib')

# Expected features (from your training)
FEATURES = ["Monto", "Edad", "H1", "H2", "H3", "H4", "H5", "H6", "MC1", "MC2", "MC3", "MC4", "MC5", "MC6", "MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "Genero_1", "Genero_2",	"Educacion_1",	"Educacion_2",	"Educacion_3",	"Educacion_4",	"Estado Civil_1",	"Estado Civil_2",	"Estado Civil_3"]

def predict_default(Monto, Genero, Educacion, Estado_Civil, Edad, H1, H2, H3, H4, H5, H6, MC1, MC2, MC3, MC4, MC5, MC6, MP1, MP2, MP3, MP4, MP5, MP6):
    # Create input data
    data = {
        'Monto': Monto, 'Edad': Edad, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'H5': H5, 'H6': H6, 'MC1': MC1, 'MC2': MC2, 'MC3': MC3, 'MC4': MC4, 'MC5': MC5, 'MC6': MC6, 'MP1': MP1, 'MP2': MP2, 'MP3': MP3, 'MP4': MP4, 'MP5': MP5, 'MP6': MP6,
        'Educacion_1': 1 if Educacion == 1 else 0,
        'Educacion_2': 1 if Educacion == 2 else 0,
        'Educacion_3': 1 if Educacion == 3 else 0,
        'Educacion_4': 1 if Educacion == 4 else 0,
        'Estado Civil_1': 1 if Estado_Civil == 1 else 0,
        'Estado Civil_2': 1 if Estado_Civil == 2 else 0,
        'Estado Civil_3': 1 if Estado_Civil == 3 else 0,
        'Genero_1': 1 if Genero == '1' else 0,
        'Genero_2': 1 if Genero == '2' else 0,

    }

    df = pd.DataFrame([data])[FEATURES]
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return "Defaulteador" if prediction == 1 else "Cumplidor", f"{probability*100:.1f}%"

if st.button("Predict"):
    prediction, probability = predict_default(Monto, Genero, Educacion, Estado_Civil, Edad, H1, H2, H3, H4, H5, H6, MC1, MC2, MC3, MC4, MC5, MC6, MP1, MP2, MP3, MP4, MP5, MP6)

    if prediction == "Defaulteador":
        st.error(f"❌ Defaulteador (Probability of default: {probability})")
    else:
        st.success(f"✅ Cumplidor (Probability of default: {probability})")

    st.text("Creado por Agustin Esteves, Ignacio Del Bianco y Bruno Petruzzi para la materia Big Data & Machine Learning (Profesor: Santiago Nuñez Rimedio).")
    st.text("Facultad de Ciencias Economicas de la Universidad de Buenos Aires.")
    
