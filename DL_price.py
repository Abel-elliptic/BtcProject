# !pip install pandas_datareader
# !pip install mplfinance
# !pip install scikit-learn
# !pip install StandardScaler
# !pip install tensorflow
from pandas_datareader import data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import mplfinance as mpf
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math
import datetime
from datetime import date
from datetime import datetime
from datetime import timedelta
%matplotlib inline
yf.pdr_override()
np.set_printoptions(threshold=0)
def getCsv(ticker, df):
  output_file_name = ticker + ".csv"
  output_path = "sample_data/csv/"+output_file_name
  df.to_csv(output_path)
  return
start = date(1966, 1, 1)
end = date.today()
ticker = "^N225"
# MACD日足
emaS=12
emaL=26
signalLine=9


spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=(2, 1, 1))
df = data.get_data_yahoo(ticker, start, end)
date = df.index
price = df['Adj Close']
# plt.figure(figsize=(20,10)).add_subplot(spec[0])
# plt.plot(date, price,label="日経225")
# plt.title("N225",size=40)
# plt.xlabel("date",size=20)
# plt.ylabel("price",size=20)
# plt.legend()


# MACD計算
df['ema_s'] = df['Close'].ewm(span=emaS).mean()
df['ema_l'] = df['Close'].ewm(span=emaL).mean()
df['macd'] = df['ema_s'] - df['ema_l']
df['signal'] = df['macd'].ewm(span=signalLine).mean()
df['diff'] = df['macd'] - df['signal']
f_plus = lambda x: x if x > 0 else 0
f_minus = lambda x: x if x < 0 else 0
df['diff+'] = df['diff'].map(f_plus)
df['diff-'] = df['diff'].map(f_minus)

# plt.figure(figsize=(20,10)).add_subplot(spec[1])
# plt.bar(date, df['Volume'], label='Volume', color='grey')
# plt.legend()
# plt.figure(figsize=(20,10)).add_subplot(spec[2])
# plt.plot(date,df[['macd']],label="MACD")
# plt.plot(date,df[['signal']],label="Signal")
# plt.bar(date,df['diff+'],color='blue')
# plt.bar(date,df['diff-'],color='red')
# plt.legend()

#目的変数となる翌日の終値を追加
df['Close_next'] = df['Adj Close'].shift(-1)
df = df[['High','Low','Open','Close','Adj Close','Volume','ema_s','ema_l','macd','signal','diff','diff+','diff-','Close_next']]
# 欠損値の削除
df = df.dropna(how='any')
#累積
n_train = int(len(df)*0.8)
n_test = int(len(df))

df_train = df[slice(0, n_train)]
df_test = df[slice(n_train, n_test)]
X_train = df_train.drop(columns=['Adj Close'])
Y_train = df_train['Adj Close']
X_test = df_test.drop(columns=['Adj Close'])
Y_test = df_test['Adj Close']

#時系列データの交差検証
valid_scores = []
#4回交差検証する
tscv = TimeSeriesSplit(n_splits=4)

for fold, (train_indics, valid_indics) in enumerate(tscv.split(X_train)):
  X_train_cv, X_valid_cv = X_train.iloc[train_indics], X_train.iloc[valid_indics]
  Y_train_cv, Y_valid_cv = Y_train.iloc[train_indics], Y_train.iloc[valid_indics]
  #LSTM構築とコンパイル関数にX_trainを渡し、変数modelに代入
  model = LinearRegression()
  #モデル学習
  model.fit(X_train_cv, Y_train_cv)
  #予測
  Y_valid_pred = model.predict(X_valid_cv)
  #予測精度の算出と表示
  score = np.sqrt(mse(Y_valid_cv, Y_valid_pred))
  # print(f'fold {fold} MAE: {score}')
  #予測精度スコアをリストに格納
  valid_scores.append(score)

print(f'valid_scores: {valid_scores}')
cv_score = np.mean(valid_scores)
print(f'cv_score: {cv_score}')

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
score = np.sqrt(mse(Y_test, Y_pred))
print(f'R_MSE: {score}')

p_start = datetime(2023, 6, 1)
today = datetime.today()
#実際のデータと予測のデータをデータフレームにまとめる
df_result = df_test[['Adj Close']]
df_result['Close_pred'] = Y_pred
# print(df_result['Adj Close'].tolist()[0])
# print(df_result['Close_pred'])
mape = pd.DataFrame()
data = []
spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=(2, 1, 1))

for i in range(len(df_result)):
  data.append(abs(df_result['Close_pred'].tolist()[i] - df_result['Adj Close'].tolist()[i]) / abs(df_result['Adj Close'].tolist()[i]))
mape['MAPE'] = data
mape['Date'] = df_result.index
mape = mape.set_index('Date')

plt.figure(figsize=(20,10)).add_subplot(spec[0])
plt.plot(df_result[['Adj Close' , 'Close_pred']])
plt.plot(df_result['Adj Close'], label = 'Adj Close', color = 'orange')
plt.plot(df_result['Close_pred'], label = 'Close_pred', color = 'blue')

plt.xlabel("Date",size=20)
plt.ylabel("JPY",size=20)
plt.xlim(p_start,today)
# print(mape)
plt.figure(figsize=(20,10)).add_subplot(spec[1])
plt.xlim(p_start,today)
plt.bar(mape.index, mape['MAPE'], label='MAPE', color='grey')
plt.xlabel("Date",size=20)
plt.ylabel("%",size=20)

plt.legend()
plt.show()

coef = pd.DataFrame(model.coef_)
coef.index = X_train.columns
b = model.intercept_
print(coef)
print(f'切片 : {b}')
print('基本統計量 ------------')
print(X_train.describe())
#説明変数の分布を揃えれば、説明変数の係数が相関を表す
