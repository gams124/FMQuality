#!/usr/bin/env python
# coding: utf-8

# In[30]:


######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from spectral import *
import spectral.io.envi as envi
import cv
import numpy as np 
import time
import numpy as np
import scipy.fftpack
import math
import spectral.io.aviris as aviris
import matplotlib.pyplot as plt
import openpyxl
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


# In[36]:


def hyper_image(file):     #Cargando una imagen espectral
    
    print("Leyendo una imagen hiperespectral")
    #img = open_image(file).load()
    img = envi.open(file) #carga la imagen hiperespectral
    
    reflactance_scale_factor = int(img.scale_factor)
    print("El factor de reflectancia es",reflactance_scale_factor)

    archivo = open(file) #lee el archivo .bil.hdr para extraer parametros 
    l=archivo.read()
    nlines = int(l[45:48])
    nsamples = int(l[59:62])
    nbands = int(l[71:74])
    
    print("El numero de lineas es", nlines)
    print("El numero de samples es", nsamples)
    print("El numero de bandas es", nbands)
    
    view = imshow(img,(120, 60, 40))
    plt.title('IMAGEN HARINA')
    print(img.shape)

    return (img, nlines, nsamples, nbands)


def spectral(img, nlines, nsamples):   #Graficando la firma espectral
    
    bands_ = img.bands.centers  #bandas espctrales
   
    #imshow(img.read_band(50))
    #plt.plot(img.read_pixel(10,10))
    
    espec = 0
    Area = nlines*nsamples
    for i in range(nlines):
        for j in range(nsamples):
            #plt.plot(img.read_pixel(i,j))
            espec = espec + img.read_pixel(i,j)
    espec = espec/Area

    plt.figure()
    plt.plot(bands_,espec)
    plt.title('FIRMA ESPECTRAL')
    plt.ylabel('Reflectancia')
    plt.xlabel('Bandas Espectrales [nm]')
    plt.grid()
    
    return (espec, bands_)

    
    
def procesamiento(df1):
    
    sc=load('std_scaler.bin')
    data_x2 = sc.transform(df1)
    df2 = pd.DataFrame(data_x2)
    return (df2)


# In[37]:


#file_ = 'C011160163'
#ext = '.bil.hdr'
#file = file_ + ext
#[img, nlines,nsamples, nbands] = hyper_image(file)
#[espec, bands] = spectral(img, nlines, nsamples)


# In[ ]:


from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
import numpy as np
from tempfile import NamedTemporaryFile
import time
import streamlit as st
from PIL import Image
from skimage.transform import resize


# Path del modelo preentrenado
MODEL_PATH_H = 'modeloRNCross_Humedad26.h5'
MODEL_PATH_P= 'modeloRNCross_Proteina21.h5'
MODEL_PATH_G='modeloRNCross_Grasa7.h5'
MODEL_PATH_C='modeloRNCross_Ceniza24.h5'




# Se recibe la data y el modelo, devuelve la predicción
def model_prediction(entrada, modelH,modelP,modelG,modelC):
    #Prediccion de humedad
    predsH = modelH.predict(entrada)
    scH=load('scalerHUMEDAD.bin')
    predH = scH.inverse_transform(predsH)
    #Prediccion de Proteina
    predsP = modelP.predict(entrada)
    scH=load('scalerProteina.bin')
    predP = scH.inverse_transform(predsP)
    #Prediccion de Grasa
    predsG = modelG.predict(entrada)
    scH=load('scalerGrasa.bin')
    predG = scH.inverse_transform(predsG)
    #Prediccion de Ceniza
    predsC = modelC.predict(entrada)
    scH=load('scalerCeniza.bin')
    predC = scH.inverse_transform(predsC)
    return  predH , predP , predG , predC



def main():
    
    model=''

    # Se carga el modelo
    if model=='':
        modelH =load_model(MODEL_PATH_H, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
        modelP =load_model(MODEL_PATH_P, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
        modelG =load_model(MODEL_PATH_G, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
        modelC =load_model(MODEL_PATH_C, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    st.title("PREDICTOR DE PARAMETROS DE IMAGENES HIPERSPECTRALES DE HARINA DE PESCADO")
    image = Image.open('Harina.jpg')
    st.image(image, use_column_width=True)
    predictS=""
    data_file_buffer = st.file_uploader("Carga la firma espectral en formato CSV ", type=["csv"])
    # El usuario carga una imagen
    if data_file_buffer is not None:
       df=pd.read_csv(data_file_buffer,sep=';',
                           names=['394.35','396.4374','398.5248','400.6122','402.6996','404.787','406.8744','408.9618','411.0492','413.1366','415.224','417.3114','419.3988','421.4862','423.5736','425.661','427.7484','429.8358','431.9232','434.0106','436.098','438.1854','440.2728','442.3602','444.4476','446.535','448.6224','450.7098','452.7972','454.8846','456.972','459.0594','461.1468','463.2342','465.3216','467.409','469.4964','471.5838','473.6712','475.7586','477.846','479.9334','482.0208','484.1082','486.1956','488.283','490.3704','492.4578','494.5452','496.6326','498.72','500.8074','502.8948','504.9822','507.0696','509.157','511.2444','513.3318','515.4192','517.5066','519.594','521.6814','523.7688','525.8562','527.9436','530.031','532.1184','534.2058','536.2932','538.3806','540.468','542.5554','544.6428','546.7302','548.8176','550.905','552.9924','555.0798','557.1672','559.2546','561.342','563.4294','565.5168','567.6042','569.6916','571.779','573.8664','575.9538','578.0412','580.1286','582.216','584.3034','586.3908','588.4782','590.5656','592.653','594.7404','596.8278','598.9152','601.0026','603.09','605.1774','607.2648','609.3522','611.4396','613.527','615.6144','617.7018','619.7892','621.8766','623.964','626.0514','628.1388','630.2262','632.3136','634.401','636.4884','638.5758','640.6632','642.7506','644.838','646.9254','649.0128','651.1002','653.1876','655.275','657.3624','659.4498','661.5372','663.6246','665.712','667.7994','669.8868','671.9742','674.0616','676.149','678.2364','680.3238','682.4112','684.4986','686.586','688.6734','690.7608','692.8482','694.9356','697.023','699.1104','701.1978','703.2852','705.3726','707.46','709.5474','711.6348','713.7222','715.8096','717.897','719.9844','722.0718','724.1592','726.2466','728.334','730.4214','732.5088','734.5962','736.6836','738.771','740.8584','742.9458','745.0332','747.1206','749.208','751.2954','753.3828','755.4702','757.5576','759.645','761.7324','763.8198','765.9072','767.9946','770.082','772.1694','774.2568','776.3442','778.4316','780.519','782.6064','784.6938','786.7812','788.8686','790.956','793.0434','795.1308','797.2182','799.3056','801.393','803.4804','805.5678','807.6552','809.7426','811.83','813.9174','816.0048','818.0922','820.1796','822.267','824.3544','826.4418','828.5292','830.6166','832.704','834.7914','836.8788','838.9662','841.0536','843.141','845.2284','847.3158','849.4032','851.4906','853.578','855.6654','857.7528','859.8402','861.9276','864.015','866.1024','868.1898','870.2772','872.3646','874.452','876.5394','878.6268','880.7142','882.8016','884.889','886.9764','889.0638','891.1512','893.2386'])
       df1=pd.DataFrame(df, columns = ['394.35','396.4374','398.5248','400.6122','402.6996','404.787','406.8744','408.9618','411.0492','413.1366','415.224','417.3114','419.3988','421.4862','423.5736','425.661','427.7484','429.8358','431.9232','434.0106','436.098','438.1854','440.2728','442.3602','444.4476','446.535','448.6224','450.7098','452.7972','454.8846','456.972','459.0594','461.1468','463.2342','465.3216','467.409','469.4964','471.5838','473.6712','475.7586','477.846','479.9334','482.0208','484.1082','486.1956','488.283','490.3704','492.4578','494.5452','496.6326','498.72','500.8074','502.8948','504.9822','507.0696','509.157','511.2444','513.3318','515.4192','517.5066','519.594','521.6814','523.7688','525.8562','527.9436','530.031','532.1184','534.2058','536.2932','538.3806','540.468','542.5554','544.6428','546.7302','548.8176','550.905','552.9924','555.0798','557.1672','559.2546','561.342','563.4294','565.5168','567.6042','569.6916','571.779','573.8664','575.9538','578.0412','580.1286','582.216','584.3034','586.3908','588.4782','590.5656','592.653','594.7404','596.8278','598.9152','601.0026','603.09','605.1774','607.2648','609.3522','611.4396','613.527','615.6144','617.7018','619.7892','621.8766','623.964','626.0514','628.1388','630.2262','632.3136','634.401','636.4884','638.5758','640.6632','642.7506','644.838','646.9254','649.0128','651.1002','653.1876','655.275','657.3624','659.4498','661.5372','663.6246','665.712','667.7994','669.8868','671.9742','674.0616','676.149','678.2364','680.3238','682.4112','684.4986','686.586','688.6734','690.7608','692.8482','694.9356','697.023','699.1104','701.1978','703.2852','705.3726','707.46','709.5474','711.6348','713.7222','715.8096','717.897','719.9844','722.0718','724.1592','726.2466','728.334','730.4214','732.5088','734.5962','736.6836','738.771','740.8584','742.9458','745.0332','747.1206','749.208','751.2954','753.3828','755.4702','757.5576','759.645','761.7324','763.8198','765.9072','767.9946','770.082','772.1694','774.2568','776.3442','778.4316','780.519','782.6064','784.6938','786.7812','788.8686','790.956','793.0434','795.1308','797.2182','799.3056','801.393','803.4804','805.5678','807.6552','809.7426','811.83','813.9174','816.0048','818.0922','820.1796','822.267','824.3544','826.4418','828.5292','830.6166','832.704','834.7914','836.8788','838.9662','841.0536','843.141','845.2284','847.3158','849.4032','851.4906','853.578','855.6654','857.7528','859.8402','861.9276','864.015','866.1024','868.1898','870.2772','872.3646','874.452','876.5394','878.6268','880.7142','882.8016','884.889','886.9764','889.0638','891.1512','893.2386'])
       st.dataframe(df)
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción"):
         # Add a placeholder
         latest_iteration = st.empty()
         bar = st.progress(0)

         for i in range(10):
          # Update the progress bar with each iteration.
          latest_iteration.text(f'Iteration {i+1}')
          bar.progress(i*10)
          time.sleep(0.1)
         entrada=procesamiento(df)
         predictS = model_prediction(entrada, modelH,modelP,modelG,modelC)
         st.subheader('Resultados')
         st.subheader('Humedad:')
         st.write(predictS[0])
         st.subheader('Proteina:')
         st.write(predictS[1])
         st.subheader('Grasa:')
         st.write(predictS[2])
         st.subheader('Ceniza:')
         st.write(predictS[3])
         

if __name__ == '__main__':
    main()


# In[ ]:




