#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st

st.title('Electron Microscopy Analyzer')

st.subheader('This app is the property of Teсon MT.')


# In[2]:


data = st.selectbox(
    'Select the scale of photo',
    ('500nm', '1000nm'))

uploaded_file = st.file_uploader("сhoose a photo", type = 'tif' )

st.image(uploaded_file, caption='researched module')


# In[3]:


import numpy as np
import cv2
from PIL import Image

img = Image.open(uploaded_file)
img = img.save("img.jpg")

area = (0,0, 2046, 1768)
image = img.crop(area)
image = image.save("image.jpg")


# In[1]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 50)
Contours, Hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 10))
plt.imshow(gray)


# In[ ]:


#находим пористость

nn = []

coef = data/400

Spor = 0

for con in Contours:
     area = cv2.contourArea(con)
     nn.append(area)
     if area > 15 and area <10000 : Spor += area

Sph = (image.shape[0])*(image.shape[1])
P = Spor/Sph*100
P


# In[2]:


fibres = cv2.drawContours(image, Contours, -1, (0, 255, 0), 1)


# In[ ]:


#максимальная пора
Req = []

for n in nn:
    r = ((n*4/3.14)**0.5)*coef
    if r > 15 and r < 250 : Req.append(r)


R = round(max(Req),2)


# In[ ]:


#число пор больше норрита
counter = 0
for r in Req:
    if r > (1450*4/3.14)**0.5: 
        counter +=1
        
print (counter)

if counter >0:
    nor = len(Req)/counter
else: nor = 0


# In[ ]:


a = (max(Req)//10 +1)*10

bin_ranges = list(range (15, int(a)+5, 5))     
plt.figure(figsize=(10,10))
freq, bins, patches = plt.hist (Req, bins=bin_ranges, edgecolor='black')

bin_centers = np.diff(bins)*0.5 + bins[:-1]

n = 0
for fr, x, patch in zip(freq, bin_centers, patches):
  height = int(freq[n])
  plt.annotate("{}".format(height),
               xy = (x, height),             # top left corner of the histogram bar
               xytext = (0,0.5),             # offsetting label position above its bar
               textcoords = "offset points", # Offset (in points) from the *xy* value
               ha = 'center', va = 'bottom'
               )
  n = n+1


# In[ ]:


st.image(fibres, caption='found pores')
st.write('Total porousity is', P)
st.write('The max pore is', R)
st.write('The numper of pores bigger than Norrit is ', nor)


# In[ ]:


arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)

