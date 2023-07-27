import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
iris = load_iris()
#入力値と目標値を抽出
#入力値x
x = iris['data']
#目標値t
t = iris['target']
#numpy,ndarrayをtensor型に変換する
x = torch.tensor(x,dtype=torch.float32)
t = torch.tensor(t,dtype=torch.int64)
#DataLoaderを定義していく.ミニバッチ学習に必要な処理を行う
##入力値と目標値をまとめる
dataset = torch.utils.data.TensorDataset(x,t)
print("dataset")
print(dataset)
print(type(dataset))
#データセットの分割 学習データ、検証データ、テストデータに分ける.テストデータを3割、他を7割に割く
#train val test=6:2:2
n_train = int(len(dataset)*0.6)
n_val = int(len(dataset)*0.2)
n_test = len(dataset)-n_train-n_val
#ランダムに分割する
torch.manual_seed(0)
train, val, test = torch.utils.data.random_split(dataset,[n_train,n_val,n_test])
print("train")
print(train)
print(type(train))
#ここからミニバッチ学習
# バッチサイズの定義 数値に決まりはない
batch_size = 10
train_loader = torch.utils.data.DataLoader(train,batch_size,shuffle=True,drop_last=True)
print("train_loader")
print(train_loader)
print(type(train_loader))
#局所解から抜け出すためにランダムシャッフルをして目的関数もバラバラにする
#drop_last は batch_sizeで割り切れなかった余りを除去してくださいという意味
val_loader = torch.utils.data.DataLoader(val,batch_size)
test_loader = torch.utils.data.DataLoader(test,batch_size)
x, t = next(iter(train_loader))
print("x")
print(x)
print(type(x))
#ネットワークの定義を行う
#4-4-3の全結合層を定義
class Net(nn.Module):
  #使用するオブジェクトを定義
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(4,4)
    self.fc2 = nn.Linear(4,3)
    #順伝播
  def forward(self, x):
    h = self.fc1(x)
    h = F.relu(h)
    h = self.fc2(x)
    return h

torch.manual_seed(0)
net = Net()
#net
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
print("optimizer")
print(optimizer)
print(type(optimizer))
#パラメーターρの設定
#ここまででネットワークの定義
#ここから学習
batch = next(iter(train_loader))
x, t = batch
#予測値yを算出する
y = net.forward(x)
print("y")
print(y)
print(type(y))
loss = F.cross_entropy(y, t)
print("loss")
print(loss)
print(type(loss))
#次に勾配を求める
loss.backward()
#パラメーターの更新
optimizer.step()
#DLは早いからGPIで処理する
#torch.cuda.is_available()-False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#デバイスへ転送する
net.to(device)
x = x.to(device)
t = t.to(device)
#勾配の初期化
optimizer.zero_grad()
#学習ループを作成する
#エポック数1
max_epoch = 1
torch.manual_seed(0)
#モデルのインスタンス化とデバイスへの転送
net = Net().to(device)
#最適化手法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
