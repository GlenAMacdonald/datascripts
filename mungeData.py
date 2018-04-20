import pandas as pd
import numpy as np

datasetname = 'cmcdataset'

#dset = get_dset(datasetname)

def getclean_top200():
    top200 = identify_top200()
    store = select_HDFstore(datasetname)
    tlisth5 = store.get('tablelistH5')
    store.close()
    top200 = top200[top200.symbol.isin(tlisth5.index)]
    return top200

def find_times(dset):
    dflist = []
    syms = dset.index.levels[0]
    for sym in syms:
        times = pd.DataFrame(dset.loc[sym].index)
        tdiff = times.diff()
        sixmin = pd.Timedelta('0 days 00:06:00')
        lastindex = times.index[-1]
        gaps = numpy.array(tdiff.index[tdiff['timestamp'] > sixmin].tolist())
        chunkstart = times.iloc[numpy.insert(gaps,0,0,axis=0)]
        chunkend = times.iloc[numpy.append(gaps-1,lastindex)]
        chunkdf = pd.concat([chunkstart.timestamp.reset_index(drop=True),chunkend.timestamp.reset_index(drop=True)], axis = 1, keys=['start','end'])
        dflist.append(chunkdf)
    timesmultidf = pd.concat(dflist, keys=syms)
    return timesmultidf

# Pretend instantaneous Volume
def conv_vol(dset):
    syms = dset.index.levels[0]
    for sym in syms:
        volume = pd.DataFrame(dset.loc[sym].vol_24h)
        vmc = dset.loc[sym].vol_24h/dset.loc[sym].mkt_cap
        vdiff = volume.vol_24h.diff()
        volest = volavg + vdiff
    return

def conv_win_2_block(window,period,periodend):
    #window is in hours, convert to 5min blocks
    window = window*12
    #period is in days, convert to 5min blocks
    period = period*24*12
    if periodend == 0:
        periodend = 1
    else:
        periodend = periodend*24*12
    return [window,period,periodend]


# for reference - column names are: vol_24h, mkt_cap, price_btc, price_usd, change_24h
def window_dset(top200,window, column,period,periodend):
    store = select_HDFstore(datasetname)
    syms = top200.symbol
    dflist = []
    for sym in syms:
        df = store.get(sym)
        df = df.loc[~df.index.duplicated(keep='first')]
        pctchange = df[column].iloc[-(period+periodend):-periodend].pct_change()
        dflogreturn = np.log(1+pctchange)
        dflogreturn.name = sym
        dfdiffwinsum = dflogreturn.rolling(window).sum()
        dflist.append(dfdiffwinsum)
    windf = pd.concat(dflist,axis=1)
    store.close()
    return windf

def getpoints(top200,column,period,periodend):
    syms = top200.symbol
    store = select_HDFstore(datasetname)
    pchange = []
    for sym in syms:
        df = store.get(sym)
        df = df.loc[~df.index.duplicated(keep='first')]
        try:
            start = df[column].iloc[-(period+periodend)]
            end = df[column].iloc[-periodend]
            pctchange = ((end-start)/start)*100
            pchange.append(pctchange)
        except Exception as e:
            print sym, e
            pchange.append(np.nan)
    df = pd.DataFrame(pchange,index = syms)
    store.close()
    return df

def avgrnkba(column,win,per,perend):
    top200 = getclean_top200()
    [window, period, periodend] = conv_win_2_block(win, per, perend)
    windf = window_dset(top200, window, column, period, periodend)
    rankdf = windf.rank(1, ascending=False)
    mrankdf = rankdf.mean().sort_values(ascending=False).rename('mean')
    srankdf = rankdf.std().sort_values(ascending=False).rename('std')
    pchangedf = getpoints(top200,column,period,periodend)
    pchangedf.rename(columns = {0:'%chg'},inplace=True)
    crankdf = pd.concat([mrankdf, srankdf, pchangedf], axis=1).sort_values(by=['mean'], ascending=False)
    return crankdf


top200 = getclean_top200()
# Length of the window - in hours
window = 12
# Length of the analysis duration - in days
period = 21
# How close to the end of the current data to stop the analysis (last update = 0) - in days
periodend = 0
# what columns should the analysis happen over
column = 'price_usd'





#Graveyard
'''
def cor24h(column,period):
    #period is in days, convertt o 5min blocks
    period = period*24*12
    store = select_HDFstore(datasetname)
    tlisth5 = store.get('tablelistH5')
    syms = tlisth5.index
    dflist = []
    for sym in syms:
        df = store.get(sym)
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df[column].iloc[-period:]
        df.name = sym
        dflist.append(df)
    df = pd.concat(dflist, axis=1)
    df = df.corr(method='pearson')
    store.close()
    return df

def corrheatmap(df):
    #This doesn't work with backend = tk
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet',30)
    cax = ax1.imshow(df.corr(), interpolation='nearest',cmap = cmap)
    ax1.grid(True)
    plt.title('Last weeks data correlation')
    labels = df.columns
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    fig.colorbar(cax, ticks = range(-1,1,step = 0.2))
    plt.show()
    return
'''