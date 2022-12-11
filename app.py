import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# display
# st.set_page_config(layout='wide')
st.set_page_config(page_title="FarisulHaq", page_icon='icon.png')


@st.cache()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)


st.title("UAS PENDAT")

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Data BMI", "Prepocessing", "Modeling", "Implementation"])

label = ['Extremely Weak', 'Weak', 'Normal',
         'Overweight', 'Obesity', 'Extreme Obesity']
with dataframe:
    progress()
    url = "https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex"
    st.markdown(
        f'[Dataset BMI]({url})')
    st.write('Height and Weight random generated, Body Mass Index Calculated')

    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with ket:
        st.write("""
                Column
                * Gender: Male / Female
                * Height: Number(cm)
                * Weight: Number(Kg)
                * Index
                Index 
                * 0 - Extremely Weak
                * 1 - Weak
                * 2 - Normal
                * 3 - Overweight
                * 4 - Obesity
                * 5 - Extreme Obesity
                """)
    with dataset:
        dt = pd.read_csv(
            'https://raw.githubusercontent.com/farisulhaq/dataset/main/bmi-dataset.csv')
        st.dataframe(dt)
        # pisahkan fitur dan label
        X = dt.drop('Index', axis=1)
        y = dt['Index']


with preporcessing:
    progress()
    scaler = joblib.load("scaled.pkl")
    st.write('One Hot Prepocessing')
    df = pd.get_dummies(X, prefix='Gender')
    scaled = scaler.fit_transform(df)
    X = pd.DataFrame(scaled, columns=df.columns)
    st.dataframe(X)


with modeling:
    progress()
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1)
    rfc, mlpc, knc, dtc = st.tabs(
        ["RandomForestClassifier", "MLPClassifier", "KNeighborsClassifier", "DecisionTreeClassifier"])
    with rfc:
        progress()
        rf = joblib.load('Rf.pkl')
        y_pred_rf = rf.predict(X_test)
        akurasi_rf = accuracy_score(y_test, y_pred_rf)
        label_rf = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_rf}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_rf}')
        st.dataframe(label_rf)
    with mlpc:
        progress()
        mlp = joblib.load('mlp.pkl')
        y_pred_mlp = mlp.predict(X_test)
        akurasi_mlp = accuracy_score(y_test, y_pred_mlp)
        label_mlp = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_mlp}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_mlp}')
        st.dataframe(label_mlp)
    with knc:
        progress()
        knn = joblib.load('knn.pkl')

        y_pred_knn = knn.predict(X_test)
        akurasi_knn = accuracy_score(y_test, y_pred_knn)
        label_knn = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_knn}')
        st.dataframe(label_knn)
    with dtc:
        progress()
        d3 = joblib.load('d3.pkl')
        y_pred_d3 = d3.predict(X_test)
        akurasi_d3 = accuracy_score(y_test, y_pred_d3)
        label_d3 = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_d3}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_d3}')
        st.dataframe(label_d3)

with implementation:
    # height
    height = st.number_input('Tinggi', value=174)
    # weight
    weight = st.number_input('Berat', value=96)
    # gender
    gander = st.selectbox('Jenis Kelamin', ['Laki-Laki', 'Prempuan'])
    gander_female = 1 if gander == 'Prempuan' else 0
    gander_male = 1 if gander == 'Laki-Laki' else 0
    model = st.selectbox("Model", ["Random Forest"])
    data = np.array([[height, weight, gander_female, gander_male]])
    data_scaled = scaler.transform(data)
    if st.button("Predict"):
        y_imp = rf.predict(data_scaled)
        st.success(f'Data Predict = {label[y_imp[0]]}')
