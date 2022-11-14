
# Importing packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

bank_marketing_cleaned = pd.read_csv('/Users/sathu/Desktop/bank_marketing/data_clean/bank_marketing_data_cleaned.csv', index_col=None)  #  replace the path

# dynamically identifying the independent features
x_var = list(bank_marketing_cleaned.columns)
x_var.remove("subscribed")

# Splitting the dataset into test and train sets 80:20 respectively
X_train, X_test, y_train, y_test = train_test_split(bank_marketing_cleaned[x_var], bank_marketing_cleaned["subscribed"] ,test_size = 0.2, random_state = 100)

print("Data has been split into train and test sets")

# Training the model on train set
model = SVC()
model.fit(X_train, y_train)

print("Model trained! Results below : \n\n")
  
# Printing prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print("\n\n")
print("Performing a Grid search to identify best set of parameters")

# defining parameter range for GridSearch
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['poly','rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# Fitting the model for grid search
grid.fit(X_train, y_train)