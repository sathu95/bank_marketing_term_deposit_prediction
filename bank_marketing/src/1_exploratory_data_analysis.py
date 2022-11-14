
# Importing required packages
import pandas as pd
import missingno as msno
from matplotlib import style
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns

# Importing bank marketing data and displaying the first 10 rows
bank_marketing = pd.read_csv('/Users/sathu/Desktop/bank_marketing/raw_data/bank_marketing.csv') # replace path
print(bank_marketing.head(10))


# Visual plot to look for missing values in the dataset
#matplotlib inline
msno.matrix(bank_marketing)

# Getting basic stats of all the columns in the dataset
dm_describe = bank_marketing.describe(include='all')  
print(dm_describe)


# Looking for the data types of the columns
print(bank_marketing.dtypes)


# Checking for duplicated values 
print(bank_marketing.duplicated().value_counts()) 

# Dropping duplicate values from the dataset
bank_marketing = bank_marketing.drop_duplicates()
print(bank_marketing.duplicated().value_counts()) 


# Understanding the split of target column via pie chart
labels =bank_marketing['subscribed'].value_counts(sort = True).index
sizes = bank_marketing['subscribed'].value_counts(sort = True)
plt.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True, startangle=360,)
plt.title('Term deposit subscription',size = 12)
plt.show()


# Plotting histograms to check the distribution of continous variables
plt.style.use('seaborn-dark-palette')
bank_marketing.hist(bins=15, figsize=(14,10), color='darkcyan')
plt.show();


# Getting the subset of dataset with only continous variables
bank_marketing_norm = bank_marketing[['age','duration','campaign','pdays','previous','emp_var_rate','cons_price_idx',
                'cons_conf_idx','euribor3m','nr_employed']]


# Plotting to visualize the linear fit of the numerical variables
for c in bank_marketing_norm.columns[:]:
    plt.figure(figsize=(8,5))
    fig=qqplot(bank_marketing_norm[c],line='45',fit='True')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Theoretical quantiles",fontsize=15)
    plt.ylabel("Sample quantiles",fontsize=15)
    plt.title("Q-Q plot of {}".format(c),fontsize=16)
    plt.grid(True)
    plt.show()


# List of categorical variables
categorical_variables = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome', 'subscribed']

#Checking for distinct values within categorical variables and their counts
for cat_value in categorical_variables:
    print("{} \n \n".format(cat_value.upper()), bank_marketing[cat_value].unique(),'\n')
    print(bank_marketing[cat_value].value_counts())




















