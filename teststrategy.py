"""
Test a Strategy Learner.  (c) 2016 Tucker Balch
"""

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl
import marketsim
import matplotlib.pyplot as plt

#Training / in sample: January 1, 2006 to December 31 2009.
#Testing / out of sample: January 1, 2010 to December 31 2010.
#Symbols: ML4T-220, IBM, UNH, SINE_FAST_NOISE
#Starting value: $100,000
#Benchmark: Buy 500 shares on the first trading day, Sell 500 shares on the last day.

def normalize_data(df):
    return df / df.ix[0, :]

def comparision_plot(title, benchmark_portvals, q_based_trading_return_portvals):
      # draw the plot
      q_based_trading_return_normalized = normalize_data(q_based_trading_return_portvals)
      benchmark_normalized = normalize_data(benchmark_portvals)

      ax = benchmark_normalized.plot(title=title, label='Benchmark', color="black")
      q_based_trading_return_normalized.plot(label='Q Learner Based Trading', color="blue", ax=ax)

      # Add axis labels and legend
      ax.set_xlabel("Date")
      ax.set_ylabel("Performence")
      ax.legend(loc='upper left')
      plt.show()

def performence_compare(price, qlearner_based_orders, title):
    # Write the orders to a csv file
    text_file = open("qlearner_based_orders.csv", "w")
    text_file.write("Date,Symbol,Order,Shares\n")
    for qlearner_based_order in qlearner_based_orders:
        text_file.write(",".join(str(x) for x in qlearner_based_order))
        text_file.write("\n")
    text_file.close()
    qlearner_based_trading_return, qlearner_based_trading_return_portvals = marketsim.run("qlearner_based_orders.csv")
    print "Q Learner Based Trading Return = {}".format(qlearner_based_trading_return)

    # Write the orders to a csv file for benchmark
    benchmark_orders = []
    benchmark_orders.append([price.index.min().date(), "IBM", 'BUY', 500])
    benchmark_orders.append([price.index.max().date(), "IBM", 'SELL', 500])
    text_file_benchmark = open("benchmark_orders.csv", "w")
    text_file_benchmark.write("Date,Symbol,Order,Shares\n")
    for benchmark_order in benchmark_orders:
        text_file_benchmark.write(",".join(str(x) for x in benchmark_order))
        text_file_benchmark.write("\n")
    text_file_benchmark.close()
    benchmark_return, benchmark_portvals = marketsim.run("benchmark_orders.csv")
    print "Benchmark Return = {}".format(benchmark_return)

    performence_vs_benchmark = (qlearner_based_trading_return - benchmark_return) * 100

    print "QLearner Based Trading outperform the Benchmark by {}%".format(performence_vs_benchmark)

    comparision_plot(title=title, benchmark_portvals= benchmark_portvals, q_based_trading_return_portvals=qlearner_based_trading_return_portvals)


def test_code(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "GOOG"
    stdate =dt.datetime(2008,1,1)
    enddate =dt.datetime(2008,1,15) # just a few days for "shake out"

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, \
        ed = enddate, sv = 10000) 

    # set parameters for testing
    sym = "IBM"
    stdate =dt.datetime(2009,1,1)
    enddate =dt.datetime(2009,1,15)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner
    df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 10000)

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=500] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 500 shares)"

    if verb: print df_trades


# Training / in sample: January 1, 2006 to December 31 2009.
# Testing / out of sample: January 1, 2010 to December 31 2010.
# Symbols: ML4T-220, IBM, UNH, SINE_FAST_NOISE

def test_code_ML4T_220_InSample(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "ML4T-220"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # set parameters for testing
    sym = "ML4T-220"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner
    #df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)
    df_trades, qlearner_based_orders = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=500] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 500 shares)"

    if verb: print df_trades
    #print df_trades
    performence_compare(prices, qlearner_based_orders, "ML4T_220 In Sample")

def test_code_ML4T_220_OutSample(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "ML4T-220"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # set parameters for testing
    sym = "ML4T-220"
    stdate =dt.datetime(2010,1,1)
    enddate =dt.datetime(2010,12,31)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner
    #df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)
    df_trades, qlearner_based_orders = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=500] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 500 shares)"

    if verb: print df_trades
    #print df_trades
    performence_compare(prices, qlearner_based_orders, "ML4T_220 Out of Sample")


def test_code_IBM_InSample(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "IBM"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # set parameters for testing
    sym = "IBM"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner
    #df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)
    df_trades, qlearner_based_orders = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=500] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 500 shares)"

    if verb: print df_trades
    #print df_trades
    performence_compare(prices, qlearner_based_orders, "IBM In Sample")


def test_code_UNH_InSample(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "UNH"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # set parameters for testing
    sym = "UNH"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner
    #df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)
    df_trades, qlearner_based_orders = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=500] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 500 shares)"

    if verb: print df_trades
    #print df_trades
    performence_compare(prices, qlearner_based_orders, "UNH In Sample")

def test_code_SINE_FAST_NOISE_InSample(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "SINE_FAST_NOISE"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, ed = enddate, sv = 100000)

    # set parameters for testing
    sym = "SINE_FAST_NOISE"
    stdate =dt.datetime(2006,1,1)
    enddate =dt.datetime(2009,12,31)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner
    #df_trades = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)
    df_trades, qlearner_based_orders = learner.testPolicy(symbol = sym, sd = stdate, ed = enddate, sv = 100000)


    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=500] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 500 shares)"

    if verb: print df_trades
    #print df_trades

    performence_compare(prices, qlearner_based_orders, "SINE_FAST_NOISE In Sample")

if __name__=="__main__":
    #test_code(verb = False)
    print
    print "test_code_ML4T_220_InSample"
    test_code_ML4T_220_InSample(verb=False)
    print

    print "test_code_ML4T_220_OutSample"
    test_code_ML4T_220_OutSample(verb=False)
    print

    print "test_code_IBM_InSample"
    test_code_IBM_InSample(verb=False)
    print

    print "test_code_UNH_InSample"
    test_code_UNH_InSample(verb=False)
    print

    print "test_code_SINE_FAST_NOISE_InSample"
    test_code_SINE_FAST_NOISE_InSample(verb=False)
    print


