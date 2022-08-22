import csv
from distutils.command.upload import upload
from json import load
import streamlit as st #Framework
import pandas as pd #Pengolahan data seperti import data csv
import numpy as np #Untuk komputasi numerik
from PIL import Image #Untuk menampilkan gambar
import pickle #untuk memanggil model dengan type file pkl / file learning dari jupyter
from sklearn.naive_bayes import GaussianNB #Import NB - function Gaussian
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Pagar bisa digunakan untuk membuat header
st.write( """
# Klasifikasi data PNS (Web Apss) 
""")

img = Image.open('pns.jpg')
img = img.resize((600, 400))

st.image(img, use_column_width=False)
st.write("image source : sindonews.com")

#Membuat sidebar di sebelah kiri untuk input parameter
st.sidebar.header('Parameter Inputan')

#Untuk parameter inputan, bisa pilih Upload file xlsx atau input langsung parameter
upload_file = st.sidebar.file_uploader("Upload File csv", type=["csv"]) #fitur uploader file dengan type csv
if upload_file is not None: #Jika inputan tidak nol/kosong maka inputan akan membaca file csv yang diupload
    inputan = pd.read_csv(upload_file)
else: #input parameter
    def input_user(): #Membuat function untuk inputan user
        SKP = st.sidebar.slider('SKP', 24.912,57.0,43.9) #same #range 32.1 sampai 59.6 dengan isian default 43.9
        Perilaku = st.sidebar.slider('Perilaku', 16.608,39.620,17.2) #same
        #Range disesuaikan dengan min max data
        #Menyimpan function ke variable data dengan array
        #Warna orange menyesuaikan nama kolom pada data
        data = {'SKP' : SKP,
                'Perilaku' : Perilaku}
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

#Menggabungkan input_user dengan data penguin
pegawai_raw = pd.read_excel('Sidang.xlsx')
pegawai = pegawai_raw.drop(columns=['K','Nama','No','NIK','Akhir'])
#Menggabungkan dataset penguin deng inputan user
df = pd.concat([inputan, pegawai], axis=0)
#df = df.iloc[:, :-1]
#Encode untuk atribute type data string - untuk fitur ordinal - 
#encode = ['gender', 'pulau']
#for col in encode:
    #dummy = pd.get_dummies(df[col], prefix=col)
    #df = pd.concat([df, dummy], axis=1)
    #del df[col]
df = inputan[:155]
#df = df[:1] #Mengambil baris pertama untuk parameter input, keterangan label kelas, pred, prob

#Menampilkan parameter inputan
st.subheader('Parameter Inputan')
if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file xlxs untuk diupload. Saat ini memakai sample inputan, (seperti tampilan dibawah).')
    st.write(df)

#Load model NBC dari file pkl
load_model = pickle.load(open('modelTA.pkl', 'rb'))

#Menerapkan NBC
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
kelas_pegawai = np.array(['Sangat Baik', 'Baik', 'Cukup', 'Kurang', 'Sangat Kurang'])
st.write(kelas_pegawai)

st.subheader('Hasil Prediksi Klasifikasi Pegawai')
st.write(kelas_pegawai[prediksi])

st.subheader('Probabilitas Hasil Prediksi Klasifikasi Pegawai')
st.write(prediksi_proba)

img = Image.open('pns3.jpg')
img = img.resize((600, 400))

st.image(img, use_column_width=False)
st.write("image source : jawapos.com")
