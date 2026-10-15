from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import os
os.system("cls")
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("spam_dataset_v2.csv")
X_train, X_test, y_train, y_test = train_test_split(data['email'], data['label'], test_size=0.2, random_state=42)
numconverter = TfidfVectorizer(stop_words="english")
X_train_num = numconverter.fit_transform(X_train)
X_test_num = numconverter.transform(X_test)
model = LogisticRegression()
model.fit(X_train_num, y_train)
new_email = input("Enter the email to check if it is spam or not: ")
new_email_num = numconverter.transform([new_email])

predictions = model.predict(new_email_num)
if predictions[0] == 1:
    print("The email is spam.")
else:
    print("The email is not spam.")
