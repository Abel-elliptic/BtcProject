# !pip install pandas_datareader
# !pip install mplfinance
# !pip install scikit-learn
# !pip install StandardScaler
# !pip install tensorflow
from pandas_datareader import data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import mplfinance as mpf
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import datetime
from datetime import date
from datetime import datetime
from datetime import timedelta
%matplotlib inline
yf.pdr_override()
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

df = df[['High','Low','Open','Close','Adj Close','Volume','ema_s','ema_l','macd','signal','diff','diff+','diff-',]]

#カラム情報を1行ずらしたデータフレームを作成する
df_shift = df.shift(-1)
#翌日の終値と本日の終値の差分を追加する
df['delta_Close'] = df_shift['Close'] - df['Close']
#目的変数 up を追加
df['Up'] = 0
df['Up'][df['delta_Close'] > 0 ] = 1
df = df.drop('delta_Close', axis=1)

df_new = df[['Open','High','Low','Close']]
# df_new.plot(kind='line')

#累積
n_train = int(len(df)*0.8)
n_val = int(len(df))
# n_test = len(df)

df_train = df[slice(0, n_train)]
df_val = df[slice(n_train, n_val)]
# df_test = df[slice(n_val, n_test)]
X_train = df_train[['Adj Close','macd','signal']]
Y_train = df_train['Up']
X_val = df_val[['Adj Close','macd','signal']]
Y_val = df_val['Up']
# X_test = df_test[['Adj Close','macd','signal']]
# Y_test = df_test['Up']
# print(X_train.index.size)
# print(Y_train.index.size)

X_train['Adj Close'].plot(kind='line')
X_val['Adj Close'].plot(kind='line')
# X_test['Adj Close'].plot(kind='line')
# plt.legend(['X_train','X_val','X_test'])
# plt.show()

from sklearn.preprocessing import StandardScaler
#標準化
def std_to_np(df):
  df_list = []
  scl = StandardScaler()
  df_std = scl.fit_transform(df)
  df_list.append(df)
  return np.array(df_list[0])
X_train_np_array = std_to_np(X_train)
X_val_np_array = std_to_np(X_val)
# X_test_np_array = std_to_np(X_test)
Y_train = Y_train.to_numpy()
Y_val = Y_val.to_numpy()
# Y_test = Y_test.to_numpy()
# print(Y_train.size)
# print(Y_train)
# print(Y_val.shape)
# print(Y_test.shape)

#LSTM構築とコンパイル関数
def lstm_comp(df):
  # 入力層/中間層/出力層のネットワークを構築
  model = Sequential()
  model.add(LSTM(256, activation='relu',input_shape=(df.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='relu'))
  #ネットワークのコンパイル
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

#時系列データの交差検証
valid_scores = []
#4回交差検証する
tscv = TimeSeriesSplit(n_splits=4)
np.set_printoptions(threshold=0)
Y_train_new = Y_train
for fold, (train_indics, valid_indics) in enumerate(tscv.split(X_train_np_array)):
  X_train, X_valid = X_train_np_array[train_indics], X_train_np_array[valid_indics]
  Y_train, Y_valid = Y_train_new[train_indics], Y_train_new[valid_indics]
  #LSTM構築とコンパイル関数にX_trainを渡し、変数modelに代入
  model = lstm_comp(X_train)
  #モデル学習
  model.fit(X_train, Y_train, epochs=10, batch_size=64)
  #予測
  Y_valid_pred = model.predict(X_valid)
  #予測結果の2値化
  Y_valid_pred = np.where(Y_valid_pred < 0.5 , 0, 1)
  #予測精度の算出と表示
  score = accuracy_score(Y_valid, Y_valid_pred)
  print(f'fold {fold} MAE: {score}')
  #予測精度スコアをリストに格納
  valid_scores.append(score)

print(f'valid_scores: {valid_scores}')
cv_score = np.mean(valid_scores)
print(f'cv_score: {cv_score}')

model = lstm_comp(X_train_np_array)
result = model.fit(X_train_np_array, Y_train_new, epochs=10, batch_size=64)
pred = model.predict(X_val_np_array)
pred = np.where(pred < 0.5 , 0, 1)
print('accuracy = ', accuracy_score(y_true = Y_val ,y_pred = pred))
for i in range(len(Y_val)):
  print(Y_val[i], pred[i][0], end=" ")

#混同行列を表示
cm = confusion_matrix(Y_val, pred)
cmp = ConfusionMatrixDisplay(cm)
cmp.plot(cmap = plt.cm.Reds)
