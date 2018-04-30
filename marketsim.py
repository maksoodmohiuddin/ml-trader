"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
import math
from util import get_data, plot_data



def get_prices(star_date, end_date, symbols, debug):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(star_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY

    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    if debug:
        print 'df_prices before cash added'
        print prices
        print

    return prices, prices_SPY


def get_trades(df_prices, df_orders, debug):
    df_trades = pd.DataFrame(data=0.0, columns=df_prices.columns, index=df_prices.index)

    for index, row in df_orders.iterrows():
        symbol_price = df_prices.get_value(index=index, col=row['Symbol'])
        symbol_current_shares = df_trades.get_value(index=index, col=row['Symbol'])
        current_cash = df_trades.get_value(index=index, col='Cash')
        if row['Order'] == 'BUY':
            df_trades.set_value(index=index, col=row['Symbol'], value=symbol_current_shares + row['Shares'])
            df_trades.set_value(index=index, col='Cash', value=current_cash + (row['Shares'] * symbol_price * -1))
        else:
            df_trades.set_value(index=index, col=row['Symbol'], value=symbol_current_shares + (row['Shares'] * -1))
            df_trades.set_value(index=index, col='Cash', value=current_cash + (row['Shares'] * symbol_price * 1))

    if debug:
        print 'df_trades'
        print df_trades
        print

    return df_trades


def get_holdings(df_trades, symbols, start_value, debug):
    df_holdings = pd.DataFrame(data=0.0, columns=df_trades.columns, index=df_trades.index)

    last_symbol_holding = {}
    for symbol in symbols:
        last_symbol_holding[symbol] = 0

    last_cash = start_value

    for index, row in df_trades.iterrows():
        for symbol in symbols:
            df_holdings.set_value(index=index, col=symbol, value=last_symbol_holding[symbol] + row[symbol])
            last_symbol_holding[symbol] = last_symbol_holding[symbol] + row[symbol]

        df_holdings.set_value(index=index, col='Cash', value=row['Cash'] + last_cash)
        last_cash = row['Cash'] + last_cash

    if debug:
        print 'df_holdings'
        print df_holdings
        print

    return df_holdings


def get_values(df_prices, df_holdings, debug):
    df_values = df_prices * df_holdings

    if debug:
        print 'df_values'
        print df_values
        print

    return df_values


def get_values_leverage(df_prices, df_holdings, debug):
    df_values = df_prices * df_holdings

    col_list = list(df_values)
    col_list.remove('Cash')
    daily_values_abs = df_values[col_list].sum(axis=1).abs()

    daily_values = df_values.sum(axis=1)

    # determine leverage violations
    df_leverage = daily_values_abs / daily_values
    df_leverage = df_leverage[df_leverage > 3.0]

    leverage_exceeded_index = None

    # leverage violation found
    if df_leverage.empty is False:
        # identify the  first trade date with leverage violation
        leverage_exceeded_index = df_leverage.index[0]

    if debug:
        print 'df_values'
        print df_values
        print

    return df_values, leverage_exceeded_index


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # to toggle debug and release/final version
    debug = False
    enforce_leverage_rule = False

    # first read the order file into a panda data frame
    df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])

    # for extra point, ignore June 15, 2011 date:
    #df_orders = df_orders[df_orders.index != '2011-06-15']

    # sort the order, just in case
    df_orders = df_orders.sort_index(0)

    if debug:
        print 'df_orders'
        print df_orders
        print

    # Scan the orders file to determine:
    # 1. Start Date
    # 2. End Date
    # 3. Symbols

    start_date = df_orders.index.min()
    end_date = df_orders.index.max()
    symbols = df_orders.Symbol.unique().tolist()

    # get prices based on order file
    prices, prices_SPY = get_prices(start_date, end_date, symbols, debug)

    #compute_price_history_benchmark(prices)

    # first make a deep copy of prices and then add 'cash' column to prices
    df_prices = prices.copy(deep=True)
    df_prices['Cash'] = pd.Series(1.0, index=df_prices.index)

    if debug:
        print 'df_prices after cash added'
        print df_prices
        print

    if enforce_leverage_rule is False:
        # calculate trades data frame
        df_trades = get_trades(df_prices, df_orders, debug)

        # calculate holdings data frame
        df_holdings = get_holdings(df_trades, symbols, start_val, debug)

        # calculate values data frame and leverage
        df_values = get_values(df_prices, df_holdings, debug)
    else:
        # loop until no leverage violations found
        calculate = True

        # re calculate if leverage rule violated for a date
        while calculate:

            # calculate trades data frame
            df_trades = get_trades(df_prices, df_orders, debug)

            # calculate holdings data frame
            df_holdings = get_holdings(df_trades, symbols, start_val, debug)

            # calculate values data frame and leverage
            df_values, leverage_exceeded_index = get_values_leverage(df_prices, df_holdings, debug)

            # leverage violations, continue
            if leverage_exceeded_index is not None:
                # remove the trade date with leverage violation
                df_orders = df_orders[df_orders.index != leverage_exceeded_index]
            # no leverage violations, exit loop
            else:
                calculate = False

    # calculate portfolio values
    daily_portfolio_values = df_values.sum(axis=1)

    if debug:
        print 'daily_portfolio_values'
        print daily_portfolio_values
        print

    return daily_portfolio_values


def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns = (daily_returns / daily_returns.shift(1)) - 1
    # ignore zeroth row:
    daily_returns = daily_returns[1:]
    return daily_returns


def compute_portfolio_stats(port_val):
    # port_val is a data frame or an ndarray of historical prices.
    # allocs: A list of allocations to the stocks, must sum to 1.0
    # rfr: The risk free return per sample period for the entire date range. We assume that it does not change.
    # sf: Sampling frequency per year

    rfr = 0.0
    sf = 252.0

    port_daily_returns = compute_daily_returns(port_val)

    # cr: Cumulative return
    cr = (port_val[-1] / port_val[0]) - 1

    # adr: Average daily return
    adr = port_daily_returns.mean()

    # sddr: Standard deviation of daily return
    sddr = port_daily_returns.std()

    # sr: Sharpe Ratio
    sr = math.sqrt(sf) * (adr / sddr)

    return cr, adr, sddr, sr


def test_code(order_file):
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = order_file
    sv = 100000

    # Process orders
    # portvals, prices_SPY, start_date, end_date = compute_portvals(orders_file = of, start_val = sv)
    portvals = compute_portvals(orders_file=of, start_val=sv)
    # start_date = portvals.index.min()
    # end_date = portvals.index.max()
    start_date = None
    end_date = None

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(prices_SPY)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    # print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
    print
    benchmark_return = (portvals[-1] - sv) / sv
    print "Benchmark Return =: {}".format( benchmark_return)

def run(order_file):

    of = order_file
    sv = 100000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)

    trading_return = (portvals[-1] - sv) / sv
    return trading_return, portvals
    #print "Benchmark Return =: {}".format(rule_based_trading_return)


if __name__ == "__main__":
    print "./orders/orders-benchmark.csv"
    test_code("./orders/orders-benchmark.csv")
    print '---'



