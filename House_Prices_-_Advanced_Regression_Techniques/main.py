import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from typing import Tuple

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data_train.drop_duplicates(['Id'])
data_test.drop_duplicates(['Id'])

# plt.scatter(data.GarageYrBlt, data.GrLivArea, c = "blue", marker = "s")
# plt.title("Looking for outliers")
# plt.xlabel("GarageYrBlt")
# plt.ylabel("GrLivArea")
# plt.show()

data_train = data_train[data_train.GrLivArea <= 4000]

y = np.log1p(data_train.loc[:, "SalePrice"])

data_train = data_train.drop(["SalePrice"], axis=1)

print(data_train.shape)
print(data_test.shape)

def fill_na(data: pd.DataFrame) -> pd.DataFrame:
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

  data.loc[:, "MSZoning"] = data.loc[:, "MSZoning"].fillna(data.MSZoning.mode()[0])
  data.loc[:, "Utilities"] = data.loc[:, "Utilities"].fillna(data.Utilities.mode()[0])
  data.loc[:, "Exterior1st"] = data.loc[:, "Exterior1st"].fillna(data.Exterior1st.mode()[0])
  data.loc[:, "Exterior2nd"] = data.loc[:, "Exterior2nd"].fillna(data.Exterior2nd.mode()[0])
  data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)
  data.loc[:, "BsmtFinSF1"] = data.loc[:, "BsmtFinSF1"].fillna(0)
  data.loc[:, "BsmtFinSF2"] = data.loc[:, "BsmtFinSF2"].fillna(0)
  data.loc[:, "TotalBsmtSF"] = data.loc[:, "TotalBsmtSF"].fillna(0)
  data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
  data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtHalfBath"].fillna(0)
  data.loc[:, "KitchenQual"] = data.loc[:, "KitchenQual"].fillna(data.KitchenQual.mode()[0])
  data.loc[:, "Functional"] = data.loc[:, "Functional"].fillna(data.Functional.mode()[0])
  data.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
  data.loc[:, "GarageArea"] = data.loc[:, "GarageArea"].fillna(0)
  data.loc[:, "SaleType"] = data.loc[:, "SaleType"].fillna(data.SaleType.mode()[0])

  print(f'Is there NaN value: {data.isnull().values.any()}')

  return data

data_train = fill_na(data_train)
data_test = fill_na(data_test)

print(data_train.shape)
print(data_test.shape)

def num_to_cat(data: pd.DataFrame) -> pd.DataFrame:
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

  return data

data_train = num_to_cat(data_train)
data_test = num_to_cat(data_test)

print(data_train.shape)
print(data_test.shape)

def cat_to_num(data: pd.DataFrame) -> pd.DataFrame:
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

  return data

data_train = cat_to_num(data_train)
data_test = cat_to_num(data_test)

print(data_train.shape)
print(data_test.shape)

def rank_impor(data: pd.DataFrame) -> pd.DataFrame:
  # corr = data.corr()
  # corr.sort_values(["SalePrice"], ascending = False, inplace = True)
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

  return data

data_train = rank_impor(data_train)
data_test = rank_impor(data_test)

print(data_train.shape)
print(data_test.shape)

test_id = data_test.loc[:, "Id"]
num_feat = data_train.select_dtypes(include = ['number']).columns.drop("Id")

data_train_num = data_train.loc[:, num_feat]
data_test_num = data_test.loc[:, num_feat]

skewness = data_train_num.apply(lambda x: skew(x))
skewed_feat = skewness[abs(skewness) > 0.5].index
data_train_num.loc[:, skewed_feat] = np.log1p(data_train_num.loc[:, skewed_feat])
data_test_num.loc[:, skewed_feat] = np.log1p(data_test_num.loc[:, skewed_feat])

def one_hot_encode(data: pd.DataFrame) -> pd.DataFrame:
  cat_feat = data.select_dtypes(exclude = ['number']).columns

  data_cat = data.loc[:, cat_feat]

  data_cat = pd.get_dummies(data_cat)

  print(f'Categorical features before align: {data_cat.columns.size}')

  return data_cat

data_train_cat = one_hot_encode(data_train)
data_test_cat = one_hot_encode(data_test)

final_data_train_cat, final_data_test_cat = data_train_cat.align(data_test_cat, join='inner', axis=1)  # inner join

print(f'Train categorical features after align: {final_data_train_cat.columns.size}')
print(f'Test categorical features after align: {final_data_test_cat.columns.size}')

data_train = pd.concat([data_train_num, final_data_train_cat], axis = 1)
data_test = pd.concat([data_test_num, final_data_test_cat], axis = 1)

X_train, X_cv, y_train, y_cv = train_test_split(data_train, y, test_size = 0.3, random_state = 0)

num_feat = data_train.select_dtypes(include = ['number']).columns

scaler = StandardScaler()
X_train.loc[:, num_feat] = scaler.fit_transform(X_train.loc[:, num_feat])
X_cv.loc[:, num_feat] = scaler.transform(X_cv.loc[:, num_feat])
X_test = data_test
X_test.loc[:, num_feat] = scaler.transform(X_test.loc[:, num_feat])

print(f'Training dataset shape: {X_train.shape}')
print(f'Cross-validation dataset shape: {X_cv.shape}')
print(f'Test dataset shape: {X_train.shape}')

_,feat_num = X_train.shape

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(feat_num,)),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.005)),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.005)),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.005)),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(.005)),
    tf.keras.layers.Dense(units=1)
])

history = model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.0005),
    metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(X_train, y_train, batch_size=50, epochs=1000)

model.evaluate(X_cv, y_cv)

y_predicted = model.predict(X_test)
y_predicted = np.expm1(y_predicted)

# for col,x1,x2,x3 in zip(X_train.columns, X_train.head(1).to_numpy().reshape(-1), X_cv.head(1).to_numpy().reshape(-1), X_test.head(1).to_numpy().reshape(-1)):
#   print(f'col: {col}, x1: {x1}, x2: {x2}, x3: {x3}')

submission = pd.DataFrame({'Id': test_id, 'SalePrice': y_predicted.reshape(-1)})
submission.to_csv('submission.csv', index=False)