#Importing required packages.
from typing import Counter
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Input data sets
DATA_PATH = "G:/.shortcut-targets-by-id/1zYOUecIzsLDVZFqSsvw6zZiBkCJY67pY/CDT_Data_Science/data_sets/Kaggle/Red Wine Quality/"
wine_df = pd.read_csv(DATA_PATH + "wineQualityReds.csv")
wine_df.columns
columns = wine_df.columns

wine_df.head()
wine_df.info()
Counter(wine_df['quality'])

#count of the target variable
sns.countplot(x='quality', data=wine_df)

##Feature Creation with Review

#next we shall create a new column called Review. This column will contain the values of 1,2, and 3. 
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
#Create an empty list called Reviews
reviews = []
for q in wine_df['quality']:
    if q >= 1 and q <= 3:
        reviews.append('1')
    elif q >= 4 and q <= 6:
        reviews.append('2')
    elif q >= 7 and q <= 10:
        reviews.append('3')
wine_df['reviews'] = reviews

#view final data
wine_df.columns
wine_df['reviews'].unique()
Counter(wine_df['reviews'])

#Now seperate the dataset x and y variables
x = wine_df.iloc[:,:11]
y = wine_df['reviews']

x.head(10)
y.head(10)

#Scale Data using StandardScalar for PCA
sc = StandardScaler()
x = sc.fit_transform(x)

#view the scaled features
print(x)

#Perform PCA
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)

#plot the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()

#AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 
#we shall pick the first 8 components for our prediction.
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)
print(x_new)




"Models for Testing"

from sklearn.model_selection import train_test_split, cross_validate
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)

#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)

#Random Forrest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)

#print confusion matrix and accuracy score
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score*100)

#NaiveBayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb_predict=nb.predict(x_test)

#print confusion matrix and accuracy score
nb_conf_matrix = confusion_matrix(y_test, nb_predict)
nb_acc_score = accuracy_score(y_test, nb_predict)
print(nb_conf_matrix)
print(nb_acc_score*100)

#SVM Classifier

from sklearn.svm import SVC
#we shall use the rbf kernel first and check the accuracy
lin_svc = SVC()
lin_svc.fit(x_train, y_train)
svm_predict=rf.predict(x_test)

#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, svm_predict)
lin_svc_acc_score = accuracy_score(y_test, svm_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)






"Confusion Matrix and Confusion Matrix Plot"
# View the classification report for test data and predictions
print(classification_report(y_test, rf_predict))

#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, rf_predict))

#### Get and reshape confusion matrix data####
matrix = confusion_matrix(y_test, rf_predict)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Bad', 'Average', 'Excelent']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for Random Forrest Model')
plt.show()
