import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Achmad Fany Fadheli ")
st.write("##### Nim   : 200411100088 ")
st.write("##### Kelas : Penambangan Data C ")
data_set_description, upload_data, preporcessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Data for Migraine Classification (Klasifikasi Migrain) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/ananthr1/weather-prediction")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. preciptation (curah hujan) :

    Semua bentuk di mana air jatuh di permukaan tanah dan badan air terbuka seperti hujan, hujan es, salju, hujan es, atau gerimis
    """)
    st.write("""2. tempmax (suhu maks) :

    Suhu Maksimum
    """)
    st.write("""3. tempmin (suhu min) :

    Suhu Minimum
    """)
    st.write("""4. wind (angin) :

    Kecepatan angin
    """)
    st.write("""5. weather (cuaca) :

    Output (keluaran)
    """)
    st.write("""Menggunakan Kolom (input) :

    precipitation
    tempmax * tempmin
    wind
    """)
    st.write("""Memprediksi kondisi cuaca (output) :

    1. drizzle (gerimis)
    2. rain (hujan)
    3. sun (matahari)
    4. snow (salju)
    5. fog (kabut)
    """)
    st.write("###### Aplikasi ini untuk : Weather Prediction (Prediksi Cuaca) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/HambaliFitrianto/Aplikasi-Web-Data-Mining-Weather-Prediction ")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6282138614807 ")
    
with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")
    df[["precipitation", "temp_max", "temp_min", "wind"]].agg(['min','max'])

    df.weather.value_counts()
    df = df.drop(columns=["date"])

    X = df.drop(columns="weather")
    y = df.weather
    "### Menghapus fitur "
    df
    X

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.weather).columns.values.tolist()

    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.weather).columns.values.tolist()
    
    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    st.write("# Implementation")
    Precipitation = st.number_input('Masukkan preciptation (curah hujan) : ')
    Temp_Max = st.number_input('Masukkan tempmax (suhu maks) : ')
    Temp_Min = st.number_input('Masukkan tempmin (suhu min) : ')
    Wind = st.number_input('Masukkan wind (angin) : ')

    def submit():
        # input
        inputs = np.array([[
            Precipitation,
            Temp_Max,
            Temp_Min,
            Wind
            ]])
        le = joblib.load("le.save")
        model1 = joblib.load("nb.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka anda prediksi cuaca : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
#         st.balloons()
        submit()
