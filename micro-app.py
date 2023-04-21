#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
st.title('Electron Microscopy Analyzer')
st.subheader('Данное приложение является собственностью ТЕКОН-МТ')

# In[2]:
#установление масштаба
scale = st.radio(
    "Выберете разрешение фотографии",
    ('1мкм', '3мкм', '5мкм', '10мкм', '30мкм'))
#загрузка фотографии и ее обработка
uploaded_file = st.file_uploader("Выберете фото ", type = 'tif' )
st.image(uploaded_file, caption='researched module')


# In[3]:


import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


img = Image.open(uploaded_file)
area = (0,0, 2046, 1768)
image = img.crop(area)
image = image.save("image.jpg")
image = cv2.imread("image.jpg")

# In[1]:
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 50)
Contours, Hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 10))
plt.imshow(gray)


# In[ ]:


#находим пористость

if scale == '1мкм' : 
    coef = 1000/400  
    a=1
elif scale =='3мкм': 
    coef = 3000/600  
    a=3
elif scale =='5мкм': 
    coef = 5000/400 
    a=5
elif scale =='10мкм': 
    coef = 10000/400
    a=10
else: coef = 30000/400 a=30
     
nn = []
Spor = 0

for con in Contours:
     area = cv2.contourArea(con)
     if area > 100/a and area <10000 : nn.append(area)

Sph = (image.shape[0])*(image.shape[1])
P = round(sum(nn)/Sph*100, 2)

# In[2]:

fibres = cv2.drawContours(image, Contours, -1, (0, 255, 0), 1)

# In[ ]:


#максимальная пора
Req = []

for n in nn:
    r = ((n*4/3.14)**0.5)*coef
    if r > 15 and r < 1000 : Req.append(r)
D = round(max(Req),2)
Dav = round(np.mean(Req), 2)

Seq = 0
for r in Req:
     if r < 35: a = 0.1
     elif r > 45: a = 2.5
     else: a = 1
     Seq += r*a 
Deq = Seq / len(Req)
     
M = round(np.median(Req),2)
# In[ ]:

Rnor = 43
#число пор больше норрита
counter = 0
for n in nn:
    if n > 1450/coef/coef : counter +=1
if counter >0: nor = round(counter/len(nn) * 100,2)
else: nor = 0        

# In[ ]:
fig, ax = plt.subplots()
a = (max(Req)//10 +1)*10

bin_ranges = list(range (15, int(a)+5, 5))     
plt.figure(figsize=(10,10))
freq, bins, patches = ax.hist (Req, bins=bin_ranges, edgecolor='black')

bin_centers = np.diff(bins)*0.5 + bins[:-1]

n = 0
for fr, x, patch in zip(freq, bin_centers, patches):
  height = int(freq[n])
  ax.annotate("{}".format(height),
               xy = (x, height),             # top left corner of the histogram bar
               xytext = (0,0.5),             # offsetting label position above its bar
               textcoords = "offset points", # Offset (in points) from the *xy* value
               ha = 'center', va = 'bottom'
               )
  n = n+1

# In[ ]:


st.image(fibres, caption='found pores')
st.write('Общая пористость образца, %', P)
st.write('Максимальный эквивалентный диаметр поры, нм', D)
st.write('Медианный эквивалентный диаметр поры, нм', M)
st.write('Средний эквивалентный диаметр поры, нм', Dav)
st.write('Доля пор больше НОРИТб %', round(nor,2))
st.pyplot(fig)

some_dict = {'porosity': [P], 'max pore': [D], 'median pore': [M], 'average pore': [Dav],'pores bigger NORIT': [nor]}
data_to_csv = pd.DataFrame.from_dict (some_dict, orient='index'). reset_index()
data= data_to_csv.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Сохранить данные",
    data=data,
    file_name='micro_res.csv',
    mime='text/csv',
)
