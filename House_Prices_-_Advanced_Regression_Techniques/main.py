import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import skew

data = pd.read_csv('train.csv')

data.drop_duplicates(['Id'])
data = data.drop(['Id'], axis=1)

# plt.scatter(data.GrLivArea, data.SalePrice, c = "blue", marker = "s")
# plt.title("Looking for outliers")
# plt.xlabel("GrLivArea")
# plt.ylabel("SalePrice")
# plt.show()

data = data[data.GrLivArea <= 4000]

y = data["SalePrice"]

print(f'Is there NaN value: {data.isnull().values.any()}')

data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)
data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna('None')
data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna('None')
data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)
data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna('None')
data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna('None')
data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna('None')
data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna('None')
data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna('None')
# Set nan to the most frequent value
data.loc[:, "Electrical"] = data.loc[:, "Electrical"].fillna(data.Electrical.mode()[0])
data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna('None')
data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna('None')
# Set nan to the mean value
data.loc[:, "GarageYrBlt"] = data.loc[:, "GarageYrBlt"].fillna(data.GarageYrBlt.mean())
data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna('None')
data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna('None')
data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna('None')
data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna('None')
data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna('None')
data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna('None')

print(f'Is there NaN value: {data.isnull().values.any()}')

print(f'Number or numerical features: {data.select_dtypes("number").columns.size}')

data = data.replace({
    "MSSubClass" : {
      20: 'C20', 30: 'C30', 40: 'C40', 45: 'C45',
      50: 'C50', 60: 'C60', 70: 'C70', 75: 'C75',
      80: 'C80', 85: 'C85', 90: 'C90', 120: 'C120',
      150: 'C150', 160: 'C160', 180: 'C180', 190: 'C190'},
    "MoSold" : {
      1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
      5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
      9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
})

print(f'Number or numerical features: {data.select_dtypes("number").columns.size}')

print(f'Numerical features: {data.select_dtypes("number").columns.size}')

data = data.replace({
    "Alley" : {"None" : 0, "Grvl" : 1, "Pave" : 2},
    "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "BsmtExposure" : { "No" : 0, "None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
    "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
    "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
    "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
    "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
    "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
    "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},
    "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
    "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
    "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
    "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
    "Street" : {"Grvl" : 1, "Pave" : 2},
    "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}
})

print(f'Numerical features: {data.select_dtypes("number").columns.size}')

cat_feat = data.select_dtypes(exclude = ['number']).columns
num_feat = data.select_dtypes(include = ['number']).columns

num_feat = num_feat.drop("SalePrice")

print(f'Categorical features: {cat_feat.size} - Numerical features: {num_feat.size}')

data_num = data[num_feat]
data_cat = data[cat_feat]

skewness = data_num.apply(lambda x: skew(x))
skewed_feat = skewness[abs(skewness) > 0.5].index
data_num[skewed_feat] = np.log1p(data_num[skewed_feat])

data_cat = pd.get_dummies(data_cat)

print(f'Categorical features: {data.select_dtypes(exclude = ["number"]).columns.size} - Numerical features: {data.select_dtypes(include = ["number"]).columns.size}')

data = pd.concat([data_num, data_cat], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.3, random_state = 0)

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(np.array(X_train))

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))

history = model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)

print(data.columns)
print(X_train.shape)
print(X_train.first)