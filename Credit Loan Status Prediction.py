#Predicting Loan Status of credit card

#import data
import pandas as pd
train = pd.read_csv('./data/samples/Credit Loan Status/credit_train.csv')

#Delete Empty rows
train = train.head(100000)

#Delete Duplicate rows
train = train.drop_duplicates(subset='Loan ID')

#Formating 'years in current job' cells
train['Years in current job'] = train['Years in current job'].replace('8 years', 8)
train['Years in current job'] = train['Years in current job'].replace('<1 year', 0.5)
train['Years in current job'] = train['Years in current job'].replace('1 year', 1)
train['Years in current job'] = train['Years in current job'].replace('2 years', 2)
train['Years in current job'] = train['Years in current job'].replace('3 years', 3)
train['Years in current job'] = train['Years in current job'].replace('4 years', 4)
train['Years in current job'] = train['Years in current job'].replace('5 years', 5)
train['Years in current job'] = train['Years in current job'].replace('6 years', 6)
train['Years in current job'] = train['Years in current job'].replace('7 years', 7)
train['Years in current job'] = train['Years in current job'].replace('9 years', 9)
train['Years in current job'] = train['Years in current job'].replace('10+ years', 11)
train['Years in current job'] = train['Years in current job'].replace('n/a', train['Years in current job'].mode())


#Finding types of Vsriables
train.dtypes

#Encoding  variables from object to categorical variables
train['Loan Status'] = train['Loan Status'].astype('category')
train['Term'] = train['Term'].astype('category')
train['Years in current job'] = train['Years in current job'].astype('category')
train['Home Ownership'] = train['Home Ownership'].astype('category')
train['Purpose'] = train['Purpose'].astype('category')

#Converting Categorical into integers
cat_columns = train.select_dtypes(['category']).columns
train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)

#Changing Incorrect Credit Score
train_data = train
for i in train_data['Credit Score'].loc[train_data['Credit Score'] > 1000]:
     train_data['Credit Score'].loc[train_data['Credit Score'] > 1000] = i/10

#Finding Missing Values
print(train_data.isnull().sum())

#Imputing Missing Values
train_data['Credit Score'] = train_data['Credit Score'].fillna(train_data['Credit Score'].mean())
train_data['Annual Income'] = train_data['Annual Income'].fillna(train_data['Annual Income'].mean())
train_data['Maximum Open Credit'] = train_data['Maximum Open Credit'].fillna(train_data['Maximum Open Credit'].mean())
#For Categorical Variables
train_data['Bankruptcies'] = train_data['Bankruptcies'].fillna(train_data['Bankruptcies'].value_counts().index[0])
train_data['Tax Liens'] = train_data['Tax Liens'].fillna(train_data['Tax Liens'].value_counts().index[0])

#Ordering the data table
train_data.columns.tolist()
train_data = train_data[[
 'Current Loan Amount',
 'Term',
 'Credit Score',
 'Annual Income',
 'Years in current job',
 'Home Ownership',
 'Purpose',
 'Monthly Debt',
 'Years of Credit History',
 'Number of Open Accounts',
 'Number of Credit Problems',
 'Current Credit Balance',
 'Maximum Open Credit',
 'Bankruptcies',
 'Tax Liens',
'Loan Status']]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data.iloc[:,:15] = sc.fit_transform(train_data.iloc[:,:15])

#Splitting Data
from sklearn.cross_validation import train_test_split
d = train_data.values
x_train, x_test, y_train, y_test = train_test_split(d[:,:15], d[:,15:], test_size = 0.25, random_state = 0)

#Model Preparation

#Model building

#Fitting model to KNN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train, y_train)

#Fitting model to kernel SVM
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'rbf', random_state = 0)
svm_classifier.fit(x_train, y_train)

#Fitting model to naive bayes
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

#Fitting model to Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(x_train, y_train)

#Fitting model to Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(x_train, y_train)

#Fitting model to xgboost
from xgboost import XGBClassifier
xg_classifier = XGBClassifier()
xg_classifier.fit(x_train, y_train)

#predicting the results
knn_pred = knn_classifier.predict(x_test)
svm_pred = svm_classifier.predict(x_test)
nb_pred = nb_classifier.predict(x_test)
dt_pred = dt_classifier.predict(x_test)
rf_pred = rf_classifier.predict(x_test)
xg_pred = xg_classifier.predict(x_test)

#validating the model with confusion matrics
knn_cm = confusion_matrix(y_test, knn_pred)     
svm_cm = confusion_matrix(y_test, svm_pred)     
nb_cm  = confusion_matrix(y_test, nb_pred)      
dt_cm = confusion_matrix(y_test, dt_pred)       
rf_cm = confusion_matrix(y_test, rf_pred)       
xg_cm = confusion_matrix(y_test, xg_pred)       

# evaluate an LDA model on the dataset using k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xg_classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

################################## Artificial Neural Network #############################################
#!pip install --upgrade --no-deps git+git://github.com/theano/theano.git
#!pip install tensorflow-gpu
#!pip install keras

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
ann_classifier = Sequential()

#Adding Input layer and hidden layer
ann_classifier.add(Dense(output_dim=8, init='uniform', activation='relu', input_dim=15))

#Adding the second hidden layer
ann_classifier.add(Dense(output_dim=8, init='uniform', activation='relu'))

#Adding the output layer
ann_classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#Compiling the ANN
ann_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting ANN to the train data
ann_classifier.fit(x_train,y_train,batch_size=15,nb_epoch=100)

#Predicting the results
ann_pred = ann_classifier.predict(x_test)
ann_pred = (ann_pred > 0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
ann_cm = confusion_matrix(y_test, ann_pred)   #77%

