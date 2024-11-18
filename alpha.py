import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# alpha = Sr - (Rf + B*(Rm - Rf))
# ticker, price, alpha, beta, VaR, sharpe
# df.sort_values(by='column_name', ascending=True, inplace=False)

def LoadData(ticker, bound='2024-01-01'):
    df = pd.read_csv(f'{ticker}.csv')[::-1]
    df = df[(df['date'] >= bound)]
    close = df['adjClose'].values
    ror = close[1:]/close[:-1] - 1.0
    price = close[-1]
    return price, close, ror

def Rate(x):
    return x[-1] / x[0] - 1.0

def Beta(x, y):
    cov = sum([(i - np.mean(x))*(j - np.mean(y)) for i, j in zip(x, y)])
    var = sum([pow(i - np.mean(x), 2) for i in x])
    return cov / var

def ValueAtRisk(x, pvalue=0.01):
    x = list(sorted(x))
    I = int(pvalue*len(x))
    return x[I]

def Sharpe(ror, rf):
    rf = pow(1 + rf, 1/252) - 1
    return (np.mean(ror) - rf)/np.std(ror)

spyPrice, spyClose, spyROR = LoadData("SPY")

stocks = ['AAPL','WMT','TSLA','GS','JPM',
          'NFLX','AMZN','ORCL','MSFT','HD','GOOGL',
          'NVDA','QCOM','JNJ']


Rf = 0.04341
Mr = Rate(spyClose)

X = []

for ticker in stocks:
    price, close, ror = LoadData(ticker)
    Sr = Rate(close)
    B = Beta(spyROR, ror)
    alpha = Sr - (Rf + B*(Mr - Rf))
    VaR = ValueAtRisk(ror)
    sharpe = Sharpe(ror, Rf)
    X.append([ticker, close[-1], alpha, B, VaR, sharpe])

pd.set_option('display.max_columns', None)

DF = pd.DataFrame(X, columns=['Stock','Price','Alpha','Beta','VaR @ 1%', 'Sharpe'])

DF = DF.sort_values(by='Alpha', ascending=False, inplace=False)

print(DF)
