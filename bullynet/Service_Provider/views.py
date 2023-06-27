
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,Tweet_Prediction_model,detection_ratio_model,detection_accuracy_model


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy_model.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_Tweet_Message_Prediction_Ratio(request):
    detection_ratio_model.objects.all().delete()
    ratio = ""
    kword = 'Hate'
    print(kword)
    obj = Tweet_Prediction_model.objects.all().filter(Q(Prediction_Type=kword))
    obj1 = Tweet_Prediction_model.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio_model.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Cyberbulling'
    print(kword1)
    obj1 = Tweet_Prediction_model.objects.all().filter(Q(Prediction_Type=kword1))
    obj11 = Tweet_Prediction_model.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio_model.objects.create(names=kword1, ratio=ratio1)

    ratio12 = ""
    kword12 = 'Non Cyberbulling'
    print(kword12)
    obj12 = Tweet_Prediction_model.objects.all().filter(Q(Prediction_Type=kword12))
    obj112 = Tweet_Prediction_model.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio_model.objects.create(names=kword12, ratio=ratio12)

    obj = detection_ratio_model.objects.all()
    return render(request, 'SProvider/Find_Tweet_Message_Prediction_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Tweet_Prediction_model.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})


def charts(request,chart_type):
    chart1 = detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Tweet_Mesage_Predict_Type(request):

    obj =Tweet_Prediction_model.objects.all()
    return render(request, 'SProvider/View_Tweet_Mesage_Predict_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Cyber_Bullying_Prediction(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Cyberbullying_Predicted_DataSets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Tweet_Prediction_model.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.Tweet_Message, font_style)
        ws.write(row_num, 1, my_row.Prediction_Type, font_style)
    wb.save(response)
    return response

def train_model(request):
    detection_accuracy_model.objects.all().delete()

    dataset = pd.read_csv('tweet_data.csv')
    dataset.head()
    dataset.info()
    dataset.describe().T

    # Preprocess Data
    def process_tweet(tweet):
        return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())

    dataset.rename(columns={'class': 'label', 'tweet': 'review'}, inplace=True)

    def apply_results(label):
        if (label == 0):
            return 0  # Hate
        elif (label == 1):
            return 1  # Bullying
        elif (label == 2):
            return 2  # Non Bullying

    dataset['results'] = dataset['label'].apply(apply_results)
    dataset.drop(['label'], axis=1, inplace=True)
    results = dataset['results'].value_counts()
    dataset.drop(['tid'], axis=1, inplace=True)

    cv = CountVectorizer()

    dataset["review"] = dataset['review'].apply(process_tweet)
    x = dataset["review"]
    y = dataset["results"]

    x = cv.fit_transform(x)
    print("Tweet")
    print(x)
    print("Label")
    print(y)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB
    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print(naivebayes)
    print(confusion_matrix(y_test, predict_nb))
    print(classification_report(y_test, predict_nb))
    models.append(('naive_bayes', NB))
    detection_accuracy_model.objects.create(names="Naive Bayes", ratio=naivebayes)

    # SVM Model
    print("SVM")
    from sklearn import svm
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    models.append(('svm', lin_clf))
    detection_accuracy_model.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    models.append(('logistic', reg))
    detection_accuracy_model.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("Decision Tree Classifier")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtcpredict = dtc.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, dtcpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, dtcpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, dtcpredict))
    detection_accuracy_model.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

    dataset.to_csv("Processed_Tweets.csv")

    obj = detection_accuracy_model.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})











