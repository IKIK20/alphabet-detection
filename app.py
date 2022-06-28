import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

x=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts())
classes= ['A', 'B', 'C', 'D','E','F','G','H','I','J','K', 'L','M','N','O','P','Q', 'R','S','T','U', 'V','W','X','Y', 'Z']
nclasses= len(classes) 

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

samples_per_class= 5
fig= plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))
idx_cls =0

for cls in classes:
  idxs=np.flatnonzero(y==cls)
  idxs= np.random.choice (idxs, samples_per_class,replace= False)
  i=0
  for idx in idxs:
    plt_idx= i*nclasses + idx_cls +1
    p= plt.subplot(samples_per_class,nclasses,plt_idx)
    p= sns.heatmap(np.reshape(x [idx],(22,30)) ,cmap=plt.cm.gray, xticklabels= False, yticklabels= False, cbar= False)
    p=plt.axis("off")
    i=i+1
  idx_cls+=1 

xtrain,xtest,ytrain,ytest= train_test_split(x,y,train_size=7500, test_size= 2500, random_state=9)
xtrainscale= xtrain/255.0
xtestscale= xtest/255.0

clf= LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrainscale,ytrain)
ypredict= clf.predict(xtestscale)
acc= accuracy_score(ytest,ypredict)
print(acc)

cm= pd.crosstab(ytest,ypredict,rownames= ["actual"],colnames= ["predicted"])
p=plt.figure(figsize=(10,10))
p=sns.heatmap(cm,annot= True, cbar= False, fmt= "d") 

cap= cv2.VideoCapture(0)
while(True):
  try:
    ret,frame= cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    height,width= grey.shape
    upperleft=(int(width/2-56),int(height/2-56))
    bottomright=(int(width/2+56),int(height/2+56))
    cv2.rectangle(grey,upperleft,bottomright,(0,255,0),2)
    roi=grey[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
    imgpil=Image.fromarray(roi)
    imgbw= imgpil.convert("L")
  except: