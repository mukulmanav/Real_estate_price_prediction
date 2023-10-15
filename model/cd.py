# Linear Regression Project
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize']=920,10
# %%
df=pd.read_csv("../Datasets/bengaluru_house_prices.csv")
# %%
df.head()
# %%
df.describe()
# %%
df.info()
# %%
df.shape
# %% to cont the distinct values in the area_type column
df['area_type'].value_counts()
# %%
df2=df.drop(['area_type','availability','society'],axis=1)
# %%
df2.head()
# %%
df2.info()
# %% Data Cleaning: Handle NA values
# to print the number of null values in all columns
df2.isnull().sum()
# %%
df3=df2.dropna()
# %%
df3.isnull().sum()
# %%
df3.head()
# %%
df3['size'].value_counts()
# %% Feature Engineering
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))
# %%
df3.head()
# %%
df3['bhk'].unique()
# %%
df3.total_sqft.value_counts()
# %%
df3[df3.bhk>20]
# %%
df3.total_sqft.unique()
# %% making a function to convert the string values into the float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
# %%
df3[df3['total_sqft'].apply(is_float)]
# %%
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   
# %%
df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
# %%
df4.head()
# %%
df4.loc[30]
# %% Feature Engineering
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()

# %% returns number of values in location column
len(df5.location.unique())
# %% 
df5.location=df5.location.apply(lambda x: x.strip()) #strip() will remove the space from the end of the value
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats
# %% Dimensionality Reduction
location_stats_less_than_10=location_stats[location_stats<=10]
location_stats_less_than_10
# %%
len(df5.location.unique())
# %% will convert the location_stats_less_than_10 values to other
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())
# %% Outlier/anamoly Removal Using Business Logic
df5[(df5.total_sqft/df5.bhk)<300].head() # it will show the value if (df5.total_sqft/df5.bhk) is <300
# %%
df6=df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape
# %%  Outlier Removal Using Standard Deviation and Mean
df6.price_per_sqft.describe()

# %%
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape
# %%
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")
# %%
plot_scatter_chart(df7,"Hebbal")
# %%
'''
We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.**

{
    '1' : {
        'mean': 4000,
        'std: 2000,
        'count': 34
    },
    '2' : {
        'mean': 4300,
        'std: 2300,
        'count': 22
    },    
}

Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment**
'''
# %% remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape
# %%
plot_scatter_chart(df8,"Rajaji Nagar")
# %%
plot_scatter_chart(df8,"Hebbal")
# %%
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
# %% Outlier Removal Using Bathrooms Feature
df8.bath.unique()
# %%
df8[df8.bath>=10]
# %%

# %%
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("No of bathroom")
plt.ylabel("count")
# %% 
df8[df8.bath>10]
# %% if number of bath > number of bedroom +2
df8[df8.bath>df8.bhk+2]
# %%
'''
if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed
'''
# %%
df9 = df8[df8.bath<df8.bhk+2]
df9.shape
# %%
df9.head(2)
#%%
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)
# %% Use One Hot Encoding For Location
dummies=pd.get_dummies(df10.location)
dummies.shape
dummies.head(5)
# %% Creating new dataframe and copying dummies data in it
df11=pd.concat([dummies.drop('other',axis='columns'),df10],axis=1)
df11.head()
# %%
df12=df11.drop('location',axis=1)
df12.head(2)
# %%
df12.shape 
# %%  Starting ML model
x=df12.drop('price',axis=1)
y=df12['price']
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)
# %%  Using linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
# %% Using K fold cross validation to measure accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)


# %% Find best model using GridSearchCV
'''
In this we are comparing all the regression techniques to Find best model using GridSearchCV
'''
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)
# %%
'''
Based on above results we can say that LinearRegression gives the best score. 
Hence we will use that.
'''
# %%
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(x.columns==location)[0][0]

    X = np.zeros(len(x.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1

    return lr.predict([X])[0]
# %%
predict_price('1st Phase JP Nagar',1000, 2, 2)
# %%
predict_price('1st Phase JP Nagar',1000, 3, 3)

# %%
predict_price('Indira Nagar',1000, 2, 2)
# %%
predict_price('Indira Nagar',1000, 3, 3)

# %%
