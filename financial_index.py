import pandas_datareader as pdr
import datetime

import numpy as np

MIN_PERIOD = 252

def annual_volatility(etf,min_periods = 252):
    # Assign `Adj Close` to `daily_close`
    daily_close = etf[['Adj Close']]

    # Daily returns
    daily_pct_change = daily_close.pct_change()

    # Replace NA values with 0
    daily_pct_change.fillna(0, inplace=True)

    # Calculate the volatility
    vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
    year=2010
    for i in range(0, 7):
        print('Year:', year+i, 'volatility:', vol[str(year+i)+'-12-30':str(year+i)+'-12-31'])

def annual_return(etf):
    # Assign `Adj Close` to `daily_close`
    daily_close = etf[['Adj Close']]
    year=2010
    for i in range(0,7):
        prices=daily_close[str(year + i) + '-1-1':str(year + i) + '-12-31']
        print('Year:', year + i, 'return:', prices.get_values()[prices.size-1]-prices.get_values()[0])


xlg = pdr.get_data_yahoo('XLG',
                          start=datetime.datetime(2010, 1, 1),
                          end=datetime.datetime(2016, 12, 31))
oef = pdr.get_data_yahoo('OEF',
                          start=datetime.datetime(2010, 1, 1),
                          end=datetime.datetime(2016, 12, 31))
iwl = pdr.get_data_yahoo('IWL',
                          start=datetime.datetime(2010, 1, 1),
                          end=datetime.datetime(2016, 12, 31))

spy = pdr.get_data_yahoo('SPY',
                          start=datetime.datetime(2010, 1, 1),
                          end=datetime.datetime(2016, 12, 31))

#print('XLG')
#vol=annual_volatility(xlg,MIN_PERIOD)

#print('OEF')
#vol=annual_volatility(oef,MIN_PERIOD)

#print('IWL')
#vol=annual_volatility(iwl,MIN_PERIOD)

print('SPY')
vol=annual_volatility(spy,MIN_PERIOD)
annual_return(spy)

