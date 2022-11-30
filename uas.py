import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler

import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Achmad Fany Fadheli ")
st.write("##### Nim   : 200411100088 ")
st.write("##### Kelas : Penambahan Data C ")
data_set_description, upload_data, preporcessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Migraine Classification (Klasifikasi Migrain) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/weinoose/migraine-classification")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. Klasifikasi Migrain :

    dataset ini disediakan untuk pengguna yang ingin mengembangkan praktik jaringan saraf mereka berdasarkan dataset numerik seperti ini. Kumpulan data ini dikumpulkan dari lebih banyak klien.
    """)
   

with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv( uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")

    X= df.drop(['Type'],axis=1)
    y= df['Type']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    X_train
    y_train
    
    #Random Forest
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred_rf= model.predict(X_test)
    y_pred_rf
    
    cm = confusion_matrix(y_test,y_pred_rf)
    print(cm)
    
    akurasi_rf = round( accuracy_score(y_test,y_pred_rf)*100)
    
    print(classification_report(y_test, y_pred_rf))
    

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)
    
    

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)  
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    rf = st.checkbox('Random Forest')
    kn = st.checkbox('K-Nearest Neighbor')
    mod = st.button("Modeling")

    
    
        
            
    #KNN
    model = KNeighborsClassifier(n_neighbors = 1)  
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    
    akurasi_kn = round(accuracy_score(y_test, predicted.round())*100)
    
    if rf :
       if mod :
           st.write('Model Random Forest accuracy score: {0:0.2f}'. format(akurasi_rf))
            
    if kn :
       if mod:
           st.write("Model K-Nearest Neighbor accuracy score : {0:0.2f}" . format(akurasi_kn))      


    import altair as alt
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi_rf,akurasi_kn],
            'Nama Model' : ['Random Forest','KNN']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)
        
    with implementation:
        st.write("# Implementation")
        Age = st.number_input('Masukkan Age (Usia) : ')
        Duration = st.number_input('Masukkan Duration (Durasi) : ')
        Frequency = st.number_input('Masukkan Frequency (Frekuensi) : ')
        Intensity = st.number_input('Masukkan Intensity (Intensitas) : ')

        def submit():
            # input
            inputs = np.array([[
                Age,
                Duration,
                Frequency,
                Intensity
                ]])
            
            le = joblib.load("le.save")
            
            if akurasi_rf > akurasi_kn:
                model = joblib.load("rf.joblib")

            elif akurasi_kn > akurasi_rf:
                model = joblib.load("kn.joblib")
                
            y_pred2 = model.predict(X, y)    
            st.write(f"Berdasarkan data yang di masukkan, maka anda prediksi migrain : {le.inverse_transform(y_pred2)[0]}")

        all = st.button("Submit")
        if all :
            st.balloons()
            submit()

        
