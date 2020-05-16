import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_error
from scipy.stats import jarque_bera
from sklearn.model_selection import train_test_split
from scipy.stats import normaltest
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse


house= pd.read_csv("train.csv")

###########################################################################################################

house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)

Y = house["SalePrice"]
X = house[["BedroomAbvGr","yeni_mi","FullBath","GarageCars"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 465)

print("Eğitim kümesindeki gözlem sayısı : {}".format(X_train.shape[0]))
print("Test kümesindeki gözlem sayısı   : {}".format(X_test.shape[0]))

X_train = sm.add_constant(X_train)

sonuclar = sm.OLS(y_train, X_train).fit()

print(sonuclar.summary())

X_test = sm.add_constant(X_test)

y_preds = sonuclar.predict(X_test)


baslik_font = {'family': 'arial','color':  'darkred','weight': 'bold','size': 15 }
eksen_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 10 }
plt.figure(dpi = 100)

plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Gerçek Değerler", fontdict=eksen_font)
plt.ylabel("Tahmin edilen Değerler", fontdict=eksen_font)
plt.title("Ücretler: Gerçek ve tahmin edilen değerler", fontdict=baslik_font)
plt.show()

print("Ortalama Mutlak Hata (MAE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_preds)))
print("Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))


# Adj R-square değerimiz şuan 0.535. Biraz yetersiz.
#################################################################################################################


# house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)
#
# Y = house["SalePrice"]
# X = house[["BedroomAbvGr","yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual","LotArea"]]
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 465)
#
# print("Eğitim kümesindeki gözlem sayısı : {}".format(X_train.shape[0]))
# print("Test kümesindeki gözlem sayısı   : {}".format(X_test.shape[0]))
#
# X_train = sm.add_constant(X_train)
#
# sonuclar = sm.OLS(y_train, X_train).fit()
#
# print(sonuclar.summary())
#
# baslik_font = {'family': 'arial','color':  'darkred','weight': 'bold','size': 15 }
# eksen_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 10 }
# X_test = sm.add_constant(X_test)
#
# y_preds = sonuclar.predict(X_test)
#
# plt.figure(dpi = 100)
# plt.scatter(y_test, y_preds)
# plt.plot(y_test, y_test, color="red")
# plt.xlabel("Gerçek Değerler", fontdict=eksen_font)
# plt.ylabel("Tahmin edilen Değerler", fontdict=eksen_font)
# plt.title("Ücretler: Gerçek ve tahmin edilen değerler", fontdict=baslik_font)
# plt.show()
#
# print("Ortalama Mutlak Hata (MSE)        : {}".format(mean_absolute_error(y_test, y_preds)))
# print("Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_preds)))
# print("Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_preds)))
# print("Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))

# Değerlerimiz daha sıklaştı bu istediğimiz şeydi .Adj. R-squared değeri de arttı.
# Ortalama Mutlak Hata (MSE)    düştü
# Ortalama Kare Hata (MSE)       düştü
# Kök Ortalama Kare Hata (RMSE)  düştü
# Ortalama Mutlak Yüzde Hata (MAPE) düştü
# Her şey istediğimiz gibi.