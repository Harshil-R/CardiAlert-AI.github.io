from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
heart_data = pd.read_csv("D:\\hackathon\\model\\templates\\heart_disease_data.csv")
heart_data.head()

# %%
heart_data.tail()

# %%
heart_data.shape

# %%
heart_data.info()

# %%
heart_data.isnull().sum()

# %%
heart_data.describe()

# %%
sns.scatterplot(x='trestbps',y='target',data=heart_data)

# %%
b=sns.barplot(x='age',y='chol',data=heart_data)

# %%
sns.barplot(x='target',y='cp',data=heart_data)

# %%
sns.barplot(x='target',y='age',data=heart_data)

# %%
sns.barplot(x='target',y='chol',data=heart_data)

# %%
#checking the distribution of target variables
heart_data['target'].value_counts()

# %%
x = heart_data.drop(columns = 'target', axis=1)
y = heart_data['target']

# %%
print(x)

# %%
print(y)

# %%
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 ,stratify = y, random_state = 35)

# %%
print(x.shape , x_train.shape , x_test.shape)

# %%
model = LogisticRegression()

# %%
model.fit(x_train , y_train)

param_grid = {'C': [0.01, 0.1, 1, 10],
              'penalty': ['l1', 'l2'],
              'solver': ['liblinear', 'saga']}
# %%
y_proba = model.predict_proba(x_test)
print(y_proba)
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

# %%
# checking the accuracy for train data
x_train_predict = model.predict(x_train)
accuracy = accuracy_score(x_train_predict, y_train)
print('Accuracy score is :',accuracy)

# %%
#checking the accuracy for test data
x_test_predict = model.predict(x_test)
test_accuracy = accuracy_score(x_test_predict, y_test)
print('Accuracy score is :',test_accuracy)

# %%
# Final deployment
arr = []


#input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)

#Create your views here.
def index(request):
    return render(request,'index.html')
def test(request):
    # Load data
    if request.method == "POST":
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        cp = request.POST.get('cp')
        trestbps = request.POST.get('trestbps')
        chol = request.POST.get('chol')
        fbs = request.POST.get('fbs')
        restecg = request.POST.get('restecg')
        thalach = request.POST.get('thalach')
        exang = request.POST.get('exang')
        oldpeak = request.POST.get('oldpeak')
        slope = request.POST.get('slope')
        ca = request.POST.get('ca')
        thal=request.POST.get('thal')
        print(age,"age",gender,"gender",cp,"cp",trestbps,"trestbps",chol,"chol",fbs,"fbs",restecg,"restecg",thalach,"thalach",exang,"exang",oldpeak,"oldpeak",slope,"slope",ca,"ca",thal,"thal")
        new_patient_pred=0
        arr=[int(age),int(gender),int(cp),int(trestbps),int(chol),int(fbs),int(restecg),int(thalach),int(exang),float(oldpeak),int(slope),int(ca),int(thal)]
        print(arr)
        input_data_array = np.asarray(arr)
        input_array_reshape = input_data_array.reshape(1,-1)
        prediction = model.predict(input_array_reshape)
        print(prediction)
        risk = ""
        new_patient_pred = best_model.predict_proba(input_array_reshape)[0][1] * 100
        if(prediction[0]==0):
            print("Person doesn't have risk of having any heart issue")
            risk="Person doesn't have risk of having any heart issue"
        else:
            print("Person have a risk of having heart problem")
            risk="Person have a risk of having heart problem"
        care=""
        if(prediction[0]!=0):
            care="You are at high risk of developing a disease. You need to take the following precautionary care :end=" ". To decrease blood hypertension, you need to take precautions on your food habits and decrease stress"

            print('_______________________________________________')
            print('You are at high risk of developing a disease. You need to take the following precautionary care :')
            print('1. To decrease blood hypertension, you need to take precautions on your food habits and decrease stress')
            print('2. To decrease cholestrol levels you need to consume HTL containing food and avoid canned food')
            print('3. To keep a check on your blood sugar, consume food having no sugar and maintain a healthy lifestyle')
            print('4. To maintain a steady heartrate perform breathing exercises and avoid anxiety')
            print('5. To count on calcium, avoid foods that forms plaques which includes food having low density lipids')
            print('6. The symptomic ECG may have blockage or weak heart muscles which leads to interupted blood flow in the heart.')
            print('   Further, testing needs to be carried out to know the exact cause')
        else:
            care="You don't need any of the precautionary care"
            print("You don't need any of the precautionary care")
        c={
            "age":age,
            "gender":gender,
            "cp":cp,
            "trestbps":trestbps,
            "chol":chol,
            "fbs":fbs,
            "restecg":restecg,
            "thalach":thalach,
            "exang":exang,
            "oldpeak":oldpeak,
            "slope":slope,
            "ca":ca,
            "thal":thal,
            "id":1,
            "risk":risk,
            "care":care,
            "predict":prediction[0],
            "new_patient":round(new_patient_pred,3)
            
        }
    else:
        c={}
    return render(request,'test.html',c)


    




