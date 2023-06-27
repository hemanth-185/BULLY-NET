
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

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('Predict_Tweet_Message_Type')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):


    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Tweet_Message_Type(request):
    if request.method == "POST":
        Tweet_Message = request.POST.get('keyword')
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
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        models.append(('DecisionTreeClassifier', dtc))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        tweet_data = [Tweet_Message]
        vector1 = cv.transform(tweet_data).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Hate'
        elif prediction == 1:
            val = 'Cyberbulling'
        elif prediction == 2:
            val = 'Non Cyberbulling'

        Tweet_Prediction_model.objects.create(Tweet_Message=Tweet_Message,Prediction_Type=val)

        return render(request, 'RUser/Predict_Tweet_Message_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Tweet_Message_Type.html')



