import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn import linear_model
from matplotlib.colors import ListedColormap

df=pd.read_csv("diabetes.csv")

#sns.countplot(x='Outcome',data=df)
#plt.show()

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=0)

x_train=np.asarray(x_train)
x_test=np.asarray(x_test)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)

scaler=Normalizer().fit(x_train)

normalized_x_train=scaler.transform(x_train)
normalized_x_test=scaler.transform(x_test)

print(normalized_x_test)

knn=KNeighborsClassifier(4)
knn.fit(normalized_x_train,y_train)
y_pred=knn.predict(normalized_x_test)

a=accuracy_score(y_test,y_pred)
print(a)

print(classification_report(y_test,y_pred))


























































































