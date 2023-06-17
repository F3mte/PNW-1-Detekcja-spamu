import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
################################
desired_width=42069
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
################################

le = LabelEncoder()
data = pd.read_csv("enron_spam_data.csv")
print(data.head)
data["Messages"] = data["Subject"] + data["Message"]
data = data.drop("Message",axis=1)
data = data.drop("Message ID",axis=1)
# print(data.head)

data['Messages'] = data['Messages'].str.lower()
data['Subject'] = data['Subject'].str.lower()

data["Messages"] = [str(x).replace(':',' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace(',',' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace('.',' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace('-',' ') for x in data["Messages"]]
data["Subject"] = [str(x).replace(':',' ') for x in data["Subject"]]
data["Subject"] = [str(x).replace(',',' ') for x in data["Subject"]]
data["Subject"] = [str(x).replace('.',' ') for x in data["Subject"]]
# print(data.head)

data["Spam/Ham"] = le.fit_transform(data["Spam/Ham"])
print(data.head)

Text = []
for line in data["Messages"]:
    words = line.split(" ")
    Text.append(words)

X_train, X_test , y_train, y_test = train_test_split(data["Messages"], data["Spam/Ham"] , test_size=0.5)
# Vectorizer = CountVectorizer()
Vectorizer = TfidfVectorizer()
count= Vectorizer.fit_transform(X_train.values)
Spam_detection = MultinomialNB()
targets = y_train.values
Spam_detection.fit(count, targets)
y_predict = Spam_detection.predict(Vectorizer.transform(X_test))
#knn
knn = KNeighborsClassifier()
knn.fit(count, targets)
y_predict_knn = knn.predict(Vectorizer.transform(X_test))

#random forests
rf = RandomForestClassifier()
rf.fit(count, targets)
y_predict_rf = rf.predict(Vectorizer.transform(X_test))

#gradient boosting
gb = GradientBoostingClassifier()
gb.fit(count, targets)
y_predict_gb = gb.predict(Vectorizer.transform(X_test))

print(accuracy_score(y_test, y_predict)) # chyba juz tego nie potrzebujemy?
# 10 krotna walidacja, 5 powtorzen, wynik nieznacznie mniejszy niż przy jednokrotnej
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=2652124)
scores = cross_val_score(Spam_detection, count, targets, scoring='accuracy', cv=rkf, n_jobs=-1)
knn_scores = cross_val_score(knn, count, targets, scoring='accuracy', cv=rkf, n_jobs=-1)
rf_scores = cross_val_score(rf, count, targets, scoring='accuracy', cv=rkf, n_jobs=-1)
gb_scores = cross_val_score(gb, count, targets, scoring='accuracy', cv=rkf, n_jobs=-1)
# wyniki dla kazdej proby
print(scores)
# usrednione wyniki
print('Naive Bayes Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('K-Nearest Neighbors Accuracy: %.3f (%.3f)' % (mean(knn_scores), std(knn_scores)))
print('Random Forest Accuracy: %.3f (%.3f)' % (mean(rf_scores), std(rf_scores)))
print('Gradient Boosting Accuracy: %.3f (%.3f)' % (mean(gb_scores), std(gb_scores)))

cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multinomial Naive Bayes Confusion Matrix")
plt.show()

cm_knn = confusion_matrix(y_test, y_predict_knn)
sns.heatmap(cm_knn, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("K-Nearest Neighbors Confusion Matrix")
plt.show()

cm_rf = confusion_matrix(y_test, y_predict_rf)
sns.heatmap(cm_rf, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

cm_gb = confusion_matrix(y_test, y_predict_gb)
sns.heatmap(cm_gb, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gradient Boosting Confusion Matrix")
plt.show()

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_predict))

print("K-Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_predict_knn))

print("Random Forest Classification Report:")
print(classification_report(y_test, y_predict_rf))

print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_predict_gb))

