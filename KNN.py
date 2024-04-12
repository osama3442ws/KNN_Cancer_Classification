import matplotlib.pyplot as plt 
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, ConfusionMatrixDisplay

from sklearn import tree
#************************************************************************************
from sklearn import preprocessing

#************************************************************************************

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster      import KMeans
from sklearn.tree         import DecisionTreeRegressor
from sklearn.tree         import DecisionTreeClassifier
from sklearn.tree         import plot_tree
from sklearn.ensemble     import RandomForestClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#************************************************************************************
#Importing the dataset

data = pd.read_csv('H:\\Programming AI\\مجلد للتطبيق و التجريب\\KNN_Cancer_Classification\\KNNAlgorithmDataset.csv')

print(data)
print(data.info())
print(data.describe())

#************************************************************************************
# Delete NULL 

data.drop("Unnamed: 32", axis=1, inplace=True)
#************************************************************************************

#sns.distplot(data["symmetry_se"],color='red')
#sns.jointplot(data=data, x= "radius_mean", y="perimeter_mean", hue="diagnosis")
#sns.countplot(x="diagnosis", data=data)
#sns.histplot(x="diagnosis", data=data)
#sns.boxplot(data=data, x="diagnosis", y="fractal_dimension_worst")

#************************************************************************************
# Preprocessing Diagnosis TRUE/FALSE
diagnosis = pd.get_dummies(data["diagnosis"], drop_first=True)

print(diagnosis)
data = pd.concat([data, diagnosis], axis=1)
data.drop("diagnosis", axis=1, inplace=True)
print(data)

scaler = StandardScaler()

scaler.fit(data.drop("M", axis=1))
#************************************************************************************
#Data Preparation: Standardize and Transform the Variable
scaler = StandardScaler()
scaler.fit(data.drop("M", axis=1))

scaled_features = scaler.transform(data.drop("M", axis=1))
print(scaled_features)

scaled_df = pd.DataFrame(data=scaled_features, columns=data.columns[:-1])
print(scaled_df)
#************************************************************************************
#Split The Dataset to Train And Test Sets

x = scaled_df
y= data["M"]

X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=101, shuffle=True)
#************************************************************************************
#Train the KNN Mode

KNN = KNeighborsClassifier(n_neighbors=1)
KNN.fit(X_train, y_train)

predictions = KNN.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#************************************************************************************
#fine tuning the KNN model for optimal K-Value
error_rate = []
for i in range(1,100):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(X_train, y_train)
    pred_i = KNN.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(18,6))
plt.plot(range(1,100), error_rate, linestyle="-.", marker="o", markerfacecolor="red", markersize=6)
plt.title("Error_Rate Vs K_Values")
plt.xlabel("K_Value")
plt.ylabel("Error_Rate")

plt.show()
