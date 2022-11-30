import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")
