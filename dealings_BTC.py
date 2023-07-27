# -*- coding: utf-8 -*-
import os
import json
import ccxt
from pprint import pprint

API_KEY = 'Jkwyqdb4cGLNDx8U'
API_SECRET = 'pfmLqSBO-C4o6Wws50kzc5yL_YK5OujF'
# /AWSアクセスキーを含むソースコードのコミットをさせないためにgit-secret


coincheck = ccxt.coincheck({'apiKey': API_KEY,'secret': API_SECRET})


# 取引通貨
def market_info():
    symbol = "BTC/JPY"
    result = dict()
    result[symbol] = coincheck.load_markets()[symbol]
    return result

# 板情報
def ticker_info():
    symbol = "BTC/JPY"
    result = dict()
    result[symbol] = coincheck.fetch_ticker(symbol=symbol)['info']
    return result

# ask       現在の売り注文の最安価格
# bid       現在の買い注文の最高価格
# last      最後の取引の価格
# high24    時間での最高取引価格
# low24     時間での最安取引価格
# volume24  時間での取引量
# timestamp 現在の時刻

# 残高
def balance_info():
    keys = ['BTC', 'JPY']
    result = dict()
    for key in keys:
        result[key] = coincheck.fetchBalance()[key]
    return result



# 引数情報
order_symbol = 'BTC/JPY'   # 取引通貨
type_value   = 'limit'     # 注文タイプ（limit:指値注文, market:成行注文）
side_value   = 'buy'       # 買い(buy) or 売り(sell)
price_value  = 4000000     # 指値価格[円/BTC]
amount_value = 0.005       # 取引数量[BTC]

# 注文コード
def Order(order_symbol, type_value, side_value, amount_value, price_value):
    order = coincheck.create_order(
        symbol = order_symbol,  # 取引通貨
        type = type_value,      # 注文タイプ
        side = side_value,       # 買 or 売
        amount = amount_value,   # 取引数量
        price =price_value,      # 指値価格
    )
    return order
# 手数料[円]
commission = 407
# 最低取引量[BTC]
min_trade_BTC = 0.005

def OrderCancel(order_id, symbol):
    result = coincheck.cancel_order(
        symbol = symbol,  # 取引通貨
        id = order_id,    # 注文ID
    )
    return result

# 引数情報
order_id = '1000000001'
# order_idの取得方法を調べる→注文時のorederのメソッドにidがある
symbol =   'BTC/JPY'


def Yakujo():
    symbol = 'BTC/JPY'
    result = coincheck.fetchMyTrades(symbol = symbol)
    return result




pprint(coincheck.fetchTradingFees())