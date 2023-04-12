import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from statsmodels.tsa.stattools import coint
from scipy.odr import *
# set the seed for the random number generator
np.random.seed(107)


def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


def fLinear(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]


def find_beta(S1, S2, use_log=False):
    '''Return the cointegration coefficient of log of S1 and S2'''
    '''Use Total Least Square method'''
    if use_log:
        data = RealData(np.log(S1), np.log(S2))
    else:
        data = RealData(S1.pct_change()[1:], S2.pct_change()[1:])

    linear = Model(fLinear)
    odr = ODR(data, linear, beta0=[0., 1.])
    beta_alpha_ = odr.run().beta
    return beta_alpha_


def beta_rolling_window_table(S1, S2, window_size=250):
    # Each rolling window is of length 'window_size'
    df = pd.DataFrame(data={'S1': S1, 'S2': S2})
    rolling_window = [df[i:i+window_size]
                      for i in range(len(df)-window_size+1)]
    beta_table = [find_beta(window['S1'], window['S2'])
                  for window in rolling_window]
    return beta_table


def ratio(S1, S2, beta, use_log=False):
    '''Computes the residual of the cointegration of prices (if use_log=False) or log prices (if use_log=True)'''
    if not use_log:
        ratio_ = beta*S1 - S2
    else:
        ratio_ = beta*np.log(S1) - np.log(S2)
    return ratio_


def z_score(ratios, ratio_mean, ratio_std):
    '''Computes z-score of the provided ratios using provided mean and std'''
    zscore_ = (ratios - ratio_mean)/ratio_std
    return zscore_


def plot_zscore(zscore, delta_plus=+1., delta_minus=-1.):
    plt.figure(figsize=(15, 7))
    zscore.plot()
    plt.axhline(0, color='black')
    plt.axhline(delta_plus, color='red', linestyle='--')
    plt.axhline(delta_minus, color='green', linestyle='--')
    plt.legend(['z-Score of beta adusted spreads',
               'Mean', f'{delta_plus}', f'{delta_minus}'])


def tradePair(S1, S2, beta, ratio_mean, ratio_std, capital=1000, stop_loss=1., use_log=False):
    '''Execute pair trading for a specified capital and stop loss'''
    '''Stop loss is expressed as a percentage of capital maximum loss'''
    '''for a specific trade'''

    # Variable declaration and initialisation
    ratios = ratio(S1, S2, beta, use_log=use_log)
    zscore = z_score(ratios, ratio_mean, ratio_std)
    # Start with no positions
    sharesS1, sharesS2 = [], []
    priceS1, priceS2 = [], []
    weightS2 = 1 / (1 + beta)
    weightS1 = beta * weightS2
    # stopLossInd will take only 3 values: -1, 0, +1
    stopLossInd = 0
    in_position = False
    dateOpenPosition, dateClosePosition = [], []
    profit_trade = []
    tradeStopLoss = -stop_loss*capital
    currentTradeProfit = 0
    dailyPnL = []

    # Main loop
    for i in range(len(ratios)):
        if (not in_position):
            dailyPnL.append(0)
            # The stop loss indicator is used to avoid taking back to back trades once stopped out
            # One scenario is that the trade is taken long at zscore -1, stopped out at zscore -2 (say),
            # immediately followed by going long again despite being still in a downtrend on the zscore
            # The indicator will be used to instruct to only take a trade once the zscore has returned
            # above -1 (in this case) or +1 in the symmtric case
            # This will reduce constantly stopped out trades in an unfavourably trending zscore
            if (abs(zscore[i]) < 1):
                stopLossInd = 0
        else:
            dailyPnL.append(sharesS1[-1]*(S1[i] - S1[i-1]) +
                            sharesS2[-1]*(S2[i] - S2[i-1]))
            currentTradeProfit = sharesS1[-1] * \
                (S1[i] - priceS1[-1]) + sharesS2[-1]*(S2[i] - priceS2[-1])

        # Define conditions for entering and exiting trades
        # This allows for easier modifications of the trading conditions
        conditionStopLoss = currentTradeProfit < tradeStopLoss
        conditionEntry1 = (not in_position) and (
            zscore[i] > 1) and (i < len(ratios)-1) and (stopLossInd != 1)

        conditionEntry2 = (not in_position) and (
            zscore[i] < -1) and (i < len(ratios)-1) and (stopLossInd != -1)

        conditionExit1 = (in_position and (abs(zscore[i]) < 0.5
                                           or i == len(ratios) - 1
                                           or conditionStopLoss))

        # Sell short if the z-score is > 1 and not last day
        if conditionEntry1:
            sharesS1.append(-weightS1 * capital / S1[i])
            sharesS2.append(weightS2 * capital / S2[i])
            priceS1.append(S1[i])
            priceS2.append(S2[i])
            dateOpenPosition.append(ratios.index[i])
            stopLossInd = 0
            in_position = True
        # Buy long if the z-score is < 1 and not last day
        elif conditionEntry2:
            sharesS1.append(weightS1 * capital / S1[i])
            sharesS2.append(-weightS2 * capital / S2[i])
            priceS1.append(S1[i])
            priceS2.append(S2[i])
            dateOpenPosition.append(ratios.index[i])
            stopLossInd = 0
            in_position = True

        # Clear positions if
        #  a) z-score between -.5 and .5 OR
        #  b) at the end of the period   OR
        #  c) Stop Loss is triggered
        elif conditionExit1:
            dateClosePosition.append(ratios.index[i])
            profit_trade.append(currentTradeProfit)
            if conditionStopLoss:
                print(f'stop loss triggered on: {ratios.index[i]}')
                stopLossInd = np.sign(zscore[i])
            in_position = False
            currentTradeProfit = 0

    # Format the output
    trades = pd.DataFrame({"profitTrade": profit_trade,
                           "dateOpenPosition": dateOpenPosition,
                           "dateClosePosition": dateClosePosition}
                          )
    pnl = pd.DataFrame(data={'PnL': dailyPnL}, index=ratios.index)
    return trades, pnl


def optimise_beta(S1, S2, stop_loss, beta_max=3, beta_min=0, use_log=False):
    ProfitMap_ = pd.DataFrame(
        columns=['profit', 'drawdown'], index=list(np.arange(beta_min, beta_max, 0.1)))
    for index, row in ProfitMap_.iterrows():
        ratios = ratio(S1, S2, index, use_log=use_log)
        ratios_mean = ratios.mean()
        ratios_std = ratios.std()
        Trade_backtest_, pnl_ = tradePair(S1, S2,
                                          beta=index,
                                          ratio_mean=ratios_mean,
                                          ratio_std=ratios_std,
                                          stop_loss=stop_loss,
                                          use_log=use_log
                                          )
        row.profit = Trade_backtest_.profitTrade.sum()
        row.drawdown = (pnl_ - pnl_.cummax()).min()

    ProfitMap_.profit = ProfitMap_.profit.astype(float)
    ProfitMap_.drawdown = ProfitMap_.drawdown.astype(float)
    return ProfitMap_


def optmise_stop_loss(S1, S2, beta, ratio_mean, ratio_std, use_log=False):
    ProfitMap_ = pd.DataFrame(
        columns=['profit', 'drawdown'], index=list(np.arange(0.0, 0.21, 0.01)))

    for index, row in ProfitMap_.iterrows():
        Trade_backtest_, pnl_ = tradePair(S1, S2,
                                          beta=beta,
                                          ratio_mean=ratio_mean,
                                          ratio_std=ratio_std,
                                          stop_loss=index,
                                          use_log=use_log
                                          )
        row.profit = Trade_backtest_.profitTrade.sum()
        row.drawdown = (pnl_ - pnl_.cummax()).min()

    ProfitMap_.profit = ProfitMap_.profit.astype(float)
    ProfitMap_.drawdown = ProfitMap_.drawdown.astype(float)
    return ProfitMap_
