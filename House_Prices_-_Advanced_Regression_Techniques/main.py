import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
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

data.loc[:, "SalePrice"] = np.log1p(data.loc[:, "SalePrice"])

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

corr = data.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
# print(corr.SalePrice)

data["OverallQual-s2"] = data["OverallQual"] ** 2
data["OverallQual-s3"] = data["OverallQual"] ** 3
data["OverallQual-Sq"] = np.sqrt(data["OverallQual"])
data["TotalBsmtSF-2"] = data["TotalBsmtSF"] ** 2
data["TotalBsmtSF-3"] = data["TotalBsmtSF"] ** 3
data["TotalBsmtSF-Sq"] = np.sqrt(data["TotalBsmtSF"])
data["1stFlrSF-2"] = data["1stFlrSF"] ** 2
data["1stFlrSF-3"] = data["1stFlrSF"] ** 3
data["1stFlrSF-Sq"] = np.sqrt(data["1stFlrSF"])
data["GrLivArea-2"] = data["GrLivArea"] ** 2
data["GrLivArea-3"] = data["GrLivArea"] ** 3
data["GrLivArea-Sq"] = np.sqrt(data["GrLivArea"])
data["BsmtQual-s2"] = data["BsmtQual"] ** 2
data["BsmtQual-s3"] = data["BsmtQual"] ** 3
data["BsmtQual-Sq"] = np.sqrt(data["BsmtQual"])
data["ExterQual-2"] = data["ExterQual"] ** 2
data["ExterQual-3"] = data["ExterQual"] ** 3
data["ExterQual-Sq"] = np.sqrt(data["ExterQual"])
data["GarageCars-2"] = data["GarageCars"] ** 2
data["GarageCars-3"] = data["GarageCars"] ** 3
data["GarageCars-Sq"] = np.sqrt(data["GarageCars"])
data["FullBath-2"] = data["FullBath"] ** 2
data["FullBath-3"] = data["FullBath"] ** 3
data["FullBath-Sq"] = np.sqrt(data["FullBath"])
data["KitchenQual-2"] = data["KitchenQual"] ** 2
data["KitchenQual-3"] = data["KitchenQual"] ** 3
data["KitchenQual-Sq"] = np.sqrt(data["KitchenQual"])
data["GarageArea-2"] = data["GarageArea"] ** 2
data["GarageArea-3"] = data["GarageArea"] ** 3
data["GarageArea-Sq"] = np.sqrt(data["GarageArea"])

cat_feat = data.select_dtypes(exclude = ['number']).columns
num_feat = data.select_dtypes(include = ['number']).columns

num_feat = num_feat.drop("SalePrice")

print(f'Categorical features: {cat_feat.size} - Numerical features: {num_feat.size}')

data_num = data.loc[:, num_feat]
data_cat = data.loc[:, cat_feat]

skewness = data_num.apply(lambda x: skew(x))
skewed_feat = skewness[abs(skewness) > 0.5].index
data_num.loc[:, skewed_feat] = np.log1p(data_num.loc[:, skewed_feat])

data_cat = pd.get_dummies(data_cat)

print(f'Categorical features: {data_cat.columns.size} - Numerical features: {data_num.columns.size}')

data = pd.concat([data_num, data_cat], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.3, random_state = 0)

scaler = StandardScaler()
X_train.loc[:, num_feat] = scaler.fit_transform(X_train.loc[:, num_feat])
X_test.loc[:, num_feat] = scaler.transform(X_test.loc[:, num_feat])

_,feat_num = X_train.shape

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(feat_num,)),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.001)),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.001)),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.001)),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.001)),
    tf.keras.layers.Dense(units=1)
])

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.0005),
    metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(X_train, y_train, batch_size=100, epochs=1000)

model.evaluate(X_test, y_test)

data_test = pd.read_csv('test.csv')S