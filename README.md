# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:  S KANUSHA SREE
RegisterNumber:  212224040149
*/
```
```
import chardet
file = "/content/spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)
```
```
import pandas as pd
data = pd.read_csv( "/content/spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())
```
```
x = data["v2"].values 
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)
```
```
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
## Output:
![image](https://github.com/user-attachments/assets/82c27f9d-0292-4374-ba74-4fba54a5b3d1)

![image](https://github.com/user-attachments/assets/45f615c0-5ec8-4f1d-8d7a-7e33f9ca3279)

![image](https://github.com/user-attachments/assets/949d00ff-ba2d-4161-ae3c-fdc9373b4343)

![image](https://github.com/user-attachments/assets/b20f6221-7083-44d2-a5e8-43b726ae04f6)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
