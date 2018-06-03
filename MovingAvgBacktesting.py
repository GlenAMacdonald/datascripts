import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.extend(['/home/Spare/CC/datascripts'])
import StoreDF
import datetime
import pandas as pd
import mungeData
import numpy as np


def get_daily_top200(window, perend, column):
    # extra day for the 6pm daily re-shuffle
    window = window + 1
    tablelist = StoreDF.get_tlisth5('cmcdataset')
    [mcp, pbtc] = get_data_range(tablelist, window, perend, column)
    daystop200 = []
    for i in mcp.index:
        dtop200 = mcp.loc[i].dropna().sort_values().iloc[-199:].index
        daystop200.append(dtop200)
    daystop200df = pd.DataFrame(daystop200,index=mcp.index)
    # Every day at 6pm(PST)/1pm(UTC) perform the rolling window calcs on the price dataframe and return the top40
    firstentry = daystop200df.index[0]
    # Get firstentry after the longest rolling window
    return firstentry, daystop200df

def get_data(sym):
    first_date = datetime.datetime.strptime('2018-03-17 22:51:00',"%Y-%m-%d %H:%M:%S")
    store = StoreDF.select_HDFstore('cmcdataset')
    df = store.get(sym)
    df = df.loc[first_date:]
    store.close()
    return df

def plot_2movavg(data1,period1,data2,period2):
    d1ema = data1.ewm(span=period1).mean()
    d1ema2 = d1ema.ewm(span=period1).mean()
    d1ema3 = d1ema2.ewm(span=period1).mean()
    d2ema = data2.ewm(span=period2).mean()
    d2ema2 = d2ema.ewm(span=period2).mean()
    d2ema3 = d2ema2.ewm(span=period2).mean()
    d1up = (data1*(d1ema3>2*d2ema3)).replace(0, np.nan)
    d1down = (data1*(d1ema3>2*d2ema3)).replace(0, np.nan)
    #d1down = (data1*~(np.logical_and(d2ema3.diff()>0,d1ema3.diff()>0))).replace(0, np.nan)
    plt.plot(d1up,color= 'g')
    plt.plot(d1down, color='r')
    return



data1 = data['price_btc']
data2 = data['vol_24h']

sym = 'MIOTA'
data = get_data(sym)
df = data
# Set the number of datapoints for the span)
period = 1.125
period2 = 5

column = 'price_btc'
pema = df[column].ewm(span=period).mean()
pema2 = pema.ewm(span=period).mean()
pema3 = pema2.ewm(span=period).mean()
pema4 = pema3.ewm(span=period).mean()
pema5 = pema4.ewm(span=period).mean()
pema6 = pema5.ewm(span=period).mean()
pema7 = pema6.ewm(span=2).mean()

pemad = df[column].diff().ewm(span=period).mean()
pema2d = pemad.ewm(span=period).mean()
pema3d = pema2d.ewm(span=period).mean()


column = 'vol_24h'
period2 = 1.5
v1ema = df[column].ewm(span=period2).mean()
v1ema2 = v1ema.ewm(span=period2).mean()
v1ema3 = v1ema2.ewm(span=period2).mean()

period3 = 2.5
v2ema = df[column].ewm(span=period3).mean()
v2ema2 = v2ema.ewm(span=period3).mean()
v2ema3 = v2ema2.ewm(span=period3).mean()


plt.plot(df[column],color='r')
plt.plot(v1ema3)
plt.plot(v2ema3)

#plt.plot(pema)
#plt.plot(pema2)
#plt.plot(pema3)
plt.plot(pema6)
plt.plot(pema7.diff())

plt.subplot(2, 1, 1)


plt.locator_params(axis='y', nbins=5)

p3 = figure(x_axis_type="datetime", plot_width=800, plot_height=350)
p3.line(y=eIMFs[-1], x=volume.index)
output_file('vimf{}.html'.format(sym))
show(p3)