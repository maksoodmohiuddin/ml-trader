"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np


class StrategyLearner(object):

    # MDP
    # States - indicators & holding
    # Actions - BUY, SELL & NOTHING
    # Reward - Daily Return

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    def normalize_data(df):
        return df / df.ix[0, :]

    def compute_daily_returns(self, df):
        """Compute and return the daily return values."""
        daily_returns = df.copy()
        daily_returns = (df / df.shift(1)) - 1
        daily_returns.ix[0,:] = 0
        return daily_returns

    def discetize_indicator(self, data, symbol):
        # copy without data
        discetize_data = pd.DataFrame(data=None, columns=data.columns,index=data.index)

        # determine discetization threshold
        data_list = data[symbol].values.tolist()
        steps = 10
        stepsize = data.shape[0]/steps
        data_list.sort()
        threshold = [None] * steps
        for i in range(0, steps):
            threshold[i] = data_list[i * stepsize]

        if self.verbose:
            print threshold

        # discetize
        for day in range(0, data.shape[0]):
            val =  data.ix[day, symbol]
            if threshold[0] <= val < threshold[1]:
                discetize_val = 0
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[1] <= val < threshold[2]:
                discetize_val = 1
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[2] <= val < threshold[3]:
                discetize_val = 2
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[3] <= val < threshold[4]:
                discetize_val = 3
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[4] <= val < threshold[5]:
                discetize_val = 4
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[5] <= val < threshold[6]:
                discetize_val = 5
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[6] <= val < threshold[7]:
                discetize_val = 6
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[7] <= val < threshold[8]:
                discetize_val = 7
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            elif threshold[8] <= val < threshold[9]:
                discetize_val = 8
                discetize_data.set_value(data.index[day], symbol, discetize_val)
            else:
                discetize_val = 9
                discetize_data.set_value(data.index[day], symbol, discetize_val)

        return discetize_data


    def discetize_states_indicators(self, symbol, sma_discetize, bbv_discetize, momentum_discetize):
        # copy without data
        discetize_states = pd.DataFrame(data=None, columns=sma_discetize.columns,index=sma_discetize.index)
        # convert the discetize indicators to a single integer
        for day in range(0, sma_discetize.shape[0]):
            X1 =  sma_discetize.ix[day, symbol]
            X2 =  bbv_discetize.ix[day, symbol]
            X3 =  momentum_discetize.ix[day, symbol]
            discetize_val = X1 + (X2 * 10) + (X3 * 100)
            discetize_states.set_value(sma_discetize.index[day], symbol, discetize_val)

        return discetize_states

    def getDiscetizeStates(self, symbol, start_date, end_date, lookback):

        symbols = [symbol]

        # since my lookback is 40, getting data beyond that will be 'safe' for the indicators, later I'll slice it
        new_start_date = start_date - pd.Timedelta(days=90)

        # Construct an appropriate DatetimeIndex object.
        dates = pd.date_range(new_start_date, end_date)

        # Read all the relevant price data (plus SPY) into a DataFrame.
        price = ut.get_data(symbols, dates)

        # Indicator 1: SMA
        # sma = price.rolling(window=lookback,min_periods=lookback).mean()
        sma = pd.rolling_mean(price, window=lookback, min_periods=lookback)

        # Indicator 2: Bollinger Bands
        #rolling_std = price.rolling(window=lookback,min_periods=lookback).std()
        rolling_std = pd.rolling_std(price, window=lookback, min_periods=lookback)
        #top_band = sma + (2 * rolling_std)
        #bottom_band = sma - (2 * rolling_std)
        bbv = (price - sma) / (2 * rolling_std)

        sma = price / sma

        # Indicator 3: momentum
        momentum = price.copy()
        momentum.values[lookback:, :] = (price.values[lookback:, :] / price.values[:-lookback, :]) - 1
        momentum.values[0:lookback, :] = np.nan

        # now slice the data to include the provided date
        price = price[start_date:end_date]
        sma = sma[start_date:end_date]
        bbv = bbv[start_date:end_date]
        momentum = momentum[start_date:end_date]

        daily_ret = self.compute_daily_returns(price)

        if self.verbose:
            print price
            print sma
            print bbv
            print momentum
            print daily_ret

        sma_discetize = self.discetize_indicator(sma, symbol)
        bbv_discetize = self.discetize_indicator(bbv, symbol)
        momentum_discetize = self.discetize_indicator(momentum, symbol)

        if self.verbose:
            print sma_discetize
            print bbv_discetize
            print momentum_discetize

        discetize_states = self.discetize_states_indicators(symbol, sma_discetize, bbv_discetize, momentum_discetize)
        return discetize_states, price


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):


        # Training / in sample: January 1, 2006 to December 31 2009.
        # Testing / out of sample: January 1, 2010 to December 31 2010.
        # Symbols: ML4T-220, IBM, UNH, SINE_FAST_NOISE
        # Starting value: $100,000
        # Benchmark: Buy 500 shares on the first trading day, Sell 500 shares on the last day.

        discetize_states, price = self.getDiscetizeStates(symbol=symbol,start_date=sd, end_date=ed, lookback=40)
        if self.verbose:
            print discetize_states

        # add your code to do learning here

        self.learner = ql.QLearner(num_states=1000,\
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.98, \
        radr = 0.999, \
        dyna = 0, \
        verbose=False) #initialize the learner

        if self.verbose:
            print self.learner.Q

        daily_returns = self.compute_daily_returns(price)
        if self.verbose:
            print daily_returns


        converged = False
        iterations = 0

        while converged is False:
            # ACTION: 0 = EXIT, 1 = GO LONG, 2 = GO SHORT

            state = discetize_states.ix[0, symbol]
            holding = 0

            iteration_reward = 0
            for day in range(0, price.shape[0]):
                day_return = daily_returns.ix[day, symbol]
                reward = holding * day_return
                iteration_reward += reward

                action = self.learner.query(state, reward)

                if action == 1:
                    if holding == 0:
                        holding = 500
                    elif holding == -500:
                        holding += 1000

                if action == 2:
                    if holding == 0:
                        holding = -500
                    elif holding == 500:
                        holding -= 1000

                state = discetize_states.ix[day, symbol]

            iterations += 1
            if self.verbose:
                print iteration_reward

            if iterations == 10:
                converged = True

        #print self.learner.Q


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        discetize_states, price = self.getDiscetizeStates(symbol=symbol,start_date=sd, end_date=ed, lookback=40)
        if self.verbose:
            print discetize_states

        qlearner_based_orders = []
        trades = price[[symbol,]]  # only portfolio symbols
        trades.values[:,:] = 0 # set them all to nothing

        state = discetize_states.ix[0, symbol]
        holding = 0

        for day in range(0, price.shape[0]):

            action = self.learner.querysetstate(state)

            if action == 1:
                if holding == 0:
                    holding = 500
                    qlearner_based_orders.append([price.index[day].date(), symbol, 'BUY', 500])
                    trades.values[day,:] = 500
                elif holding == -500:
                    holding += 1000
                    qlearner_based_orders.append([price.index[day].date(), symbol, 'BUY', 1000])
                    trades.values[day,:] = 1000

            if action == 2:
                if holding == 0:
                    holding = -500
                    qlearner_based_orders.append([price.index[day].date(), symbol, 'SELL', 500])
                    trades.values[day,:] = -500
                elif holding == 500:
                    holding -= 1000
                    qlearner_based_orders.append([price.index[day].date(), symbol, 'SELL', 1000])
                    trades.values[day,:] = -1000

            state = discetize_states.ix[day, symbol]

        return trades, qlearner_based_orders
        #return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"