#Seckin Kintas 23/05/2023

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cufflinks as cf
from plotly.offline import init_notebook_mode
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score

cf.go_offline()
init_notebook_mode(connected=True)

# Load and inspect data
telecom_df = pd.read_csv("telecom_churn.csv")
print(telecom_df.dtypes)

# Visualizations
telecom_df.hist(figsize=(30, 30))
plt.show()

plt.figure(figsize=[10, 10])
telecom_df["class"].value_counts().plot(kind='pie')
plt.show()

corr_matrix = telecom_df.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, linewidths=1, annot=True, fmt=".2f")
plt.title("Correlation Matrix of Telecom Customers", fontsize=20)
plt.show()


# Preprocessing
X = telecom_df.drop(["class", "area_code", "phone_number"], axis="columns")
y = telecom_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)

# Train models and evaluate
models = [
    ("Logistic Regression", LogisticRegression()),
    ("SVM", CalibratedClassifierCV(LinearSVC(max_iter=100000))),
    ("Random Forest", RandomForestClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("Naive bayes", GaussianNB())
]

roc_curves = []
for name, model in models:
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_predict))
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True)
    plt.show()
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=1)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"AUC score: {auc_score}")
    roc_curves.append((name, fpr, tpr))

# Plot ROC curves
plt.title('Receiver Operator Characteristics (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
for name, fpr, tpr in roc_curves:
    plt.plot(fpr, tpr, linestyle="--", label=name)
plt.legend(loc='best')
plt.savefig('ROC', dpi=300)
plt.show()


#Keep Going
