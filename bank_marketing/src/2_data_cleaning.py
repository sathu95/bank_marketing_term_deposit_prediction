
# Importing packages
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Importing bank marketing dataset
bank_marketing = pd.read_csv('/Users/sathu/Desktop/bank_marketing/raw_data/bank_marketing.csv')

# Replacing 'basic.4y','basic.6y','basic.9y' with 'mid.school'
bank_marketing['education'] = bank_marketing['education'].replace(['basic.4y','basic.6y','basic.9y'], 'mid.school') 
#print(bank_marketing['education'].value_counts())

# Getting the categorical & numerical variables in a list
categorical_variables = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome', 'subscribed']

num_var = ['age','duration','campaign','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']

print("Categorical variables : {}\n\n".format(categorical_variables))
print("Numerical variables : {}\n\n".format(num_var))


# Setting up label encoder to use it for categorical variable transformation
labelencoder_X = LabelEncoder()

# Transforming [object] categorical values into numerical values using label encoder
for feature in categorical_variables:
    bank_marketing[feature] = labelencoder_X.fit_transform(bank_marketing[feature]) 

print("Categorical values have been transformed to numerical values using Label Encoder\n\n")
    
# Applying maximum absolute scaling to normalize data
for feature in bank_marketing.columns:
    bank_marketing[feature] = bank_marketing[feature]  / bank_marketing[feature].abs().max()

print("Data has been normalized!\n\n")
      

# Checking correlation of variables
## Creating correlation matrix
corr_matrix = bank_marketing.corr().abs() 
# Selecting the upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
# Finding features with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)] 
# Droping features with high correlation
bank_marketing.drop(to_drop, axis=1, inplace=True) 

print("Highly correlated columns have been identified and dropped\n\n")


# Balancing the dataset
print("Balancing the dataset!\n\n")
print("Unbalanced Dataset size:", len(bank_marketing))
unsubscribed_data = bank_marketing[bank_marketing['subscribed'] == 0]
subscribed_data = bank_marketing[bank_marketing['subscribed'] == 1]
balanced_bank_marketing = pd.concat([unsubscribed_data.sample(len(subscribed_data), random_state=5), subscribed_data])
print("Balanced Dataset size:", len(balanced_bank_marketing))


# Performing t-test on features to check their significance over the target variable
p_value = {}
for feature in balanced_bank_marketing.columns:
    print('\n\n*** Results for {} ***\n'.format(feature))
    subscribed = balanced_bank_marketing[balanced_bank_marketing['subscribed']==1][feature]
    not_subscribed = balanced_bank_marketing[balanced_bank_marketing['subscribed']==0][feature]
    tstat, pval = stats.ttest_ind(subscribed, not_subscribed, equal_var=False)
    p_value[feature] = pval
    print('t-statistic: {:.1f}, p-value: {:.3}\n'.format(tstat, pval))
    
# Considering features with p-value less than 0.05 
column_list = []
for key, value in p_value.items():
    if value < 0.05:
        column_list.append(key)
    else:
        continue

print("Columns with p-values >= 0.05 have been dropped!\n\n")

# Getting the subset of dataset with columns that have p-values with less than 0.05
balanced_bank_marketing = balanced_bank_marketing[column_list]

# Resetting the index before saving the cleaned data as a csv
balanced_bank_marketing.reset_index(inplace = True, drop = True)
balanced_bank_marketing.to_csv("/Users/sathu/Desktop/bank_marketing/data_clean/bank_marketing_data_cleaned.csv", index=False)

print("Data has been cleaned & saved!!\n\n")











