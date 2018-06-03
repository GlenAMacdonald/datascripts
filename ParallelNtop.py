import matplotlib
import pandas as pd
import sys
sys.path.extend(['/home/Spare/CC/datascripts'])
import getdata
import mungeData
from multiprocessing import Process, Queue
import StoreDF
import datetime
import numpy as np
import copy
import itertools
import multiprocessing

# ParallelNtop aims to be the parallelized version of NTopWindow
# It loads the entire HDF5 data store into memory and uses it instead of relying on the HDD store
# This is to allow multiprocessing to occur without race access conditions - couldn't think of an easier way to do so.

def load_data_into_mem():
    store = StoreDF.select_HDFstore('cmcdataset')
    tablelist = StoreDF.get_tlisth5('cmcdataset')
    coindatalist = []
    for sym in tablelist.index:
        df = store.get(sym)
        # remove duplicates
        df = df.loc[~df.index.duplicated(keep='first')]
        # cut off any 1min data
        FiveMinStart = datetime.datetime(year=2018,month=03,day=17,hour=22,minute=51)
        df = df.loc[FiveMinStart:]
        coindatalist.append(df)
    cmcdf = pd.concat(coindatalist, keys=tablelist.index)
    return cmcdf

def get_data_range(cmcdf,winper,perend,column):
    [window, period, periodend] = mungeData.conv_win_2_block(0, winper, perend)
    lastupdate = datetime.datetime.strptime(max(tablelist.last_updated),"%Y-%m-%d %H:%M")
    if periodend == 0:
        enddate = lastupdate
    else:
        enddate = lastupdate - datetime.timedelta(minutes = 5*periodend)
    startdate = enddate - datetime.timedelta(minutes = 5*period)
    store = StoreDF.select_HDFstore('cmcdataset')
    syms = tablelist.index
    mktcap = []
    othercol = []
    for sym in syms:
        symdata = store.get(sym)
        symdata = symdata.loc[~symdata.index.duplicated(keep='first')]
        lastsymupdate = datetime.datetime.strptime(tablelist.loc[sym][0],"%Y-%m-%d %H:%M")
        if lastsymupdate > startdate:
            if lastsymupdate < enddate:
                end = lastsymupdate
            else:
                end = enddate
            #First 5min entry was at 2018-03-17 22:51
            firstentry = symdata.iloc[0].name.replace(tzinfo=None)
            if firstentry > startdate:
                start = firstentry
            else:
                start = startdate
            mktcap.append(symdata.mkt_cap.loc[start:end].rename(sym))
            coldata = symdata[[column]].loc[start:end]
            coldata.set_axis([sym], axis=1, inplace=True)
            othercol.append(coldata)
    mcp = pd.concat(mktcap,axis=1)
    ocol = pd.concat(othercol,axis=1)
    return mcp,ocol

def rollingtopN(window, perend, winX, hrX, winY, hrY, column, N1,N2,t):
    #extra day for the 6pm daily re-shuffle
    window = window + 1
    ## window = n days over which the analysis takes place,
    ## winX = n days, larger of X and Y, duration over which the first ranking window occurs
    ## winY = n days, smaller of X and Y, duration over which the second ranking window occurs
    ## hrX = n hours, hourly interval over which the rolling average window for winX occurs
    ## hrY = n hours, hourly interval over which the rolling average window for winY occurs
    ## column = dataset column used for the analysis, generally 'price_btc' or 'price_usd'
    ## N = topN
    # Re-arrange winX and winY so that winX is larger than winY
    '''if winX < winY:
        temp = winX
        winX = winY
        winY = temp
        temp = hrX
        hrX = hrY
        hyY = temp'''
    # Gets existing data for the window,
    # For each day:
    # Identifies the top 200 for the day
    # Determines average return rank (ARR) for the top 200 over the previous X days,
    # Compares it to the ARR for the top 200 over the last Y days,
    # See's which currencies are shared over both sets,
    # Removes the top 3 (usually bogus data or stupid coins) and records the symbols and dates.
    # Relies on another function to give prices for the 'buys' and 'sells'
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
    fearw = firstentry + datetime.timedelta(days = winX)
    # find all entries @ +/- 2 min from the specificed time,
    # Note: 18:00 PST = 01:00 UTC:
    tmargin = datetime.timedelta(minutes=2)
    t1 = (t + tmargin).time()
    t2 = (t - tmargin).time()
    df = daystop200df.loc[daystop200df.index.time <= t1]
    df2 = df.loc[df.index.time >= t2]
    times = df2.loc[df2.index >= fearw].index
    winXmrank = []
    winXsrank = []
    winYmrank = []
    winYsrank = []
    for time in times:
        [mrankdfX, srankdfX] = rankattime(time, winX, hrX, pbtc)
        [mrankdfY, srankdfY] = rankattime(time, winY, hrY, pbtc)
        winXmrank.append(mrankdfX)
        winXsrank.append(srankdfX)
        winYmrank.append(mrankdfY)
        winYsrank.append(srankdfY)
    winXmrankdf = pd.concat(winXmrank,axis=1)
    winXmrankdf.columns = times.strftime("%Y-%m-%d %H:%M")
    winXsrankdf = pd.concat(winXsrank, axis=1)
    winXsrankdf.columns = times.strftime("%Y-%m-%d %H:%M")
    winYmrankdf = pd.concat(winYmrank, axis=1)
    winYmrankdf.columns = times.strftime("%Y-%m-%d %H:%M")
    winYsrankdf = pd.concat(winYsrank, axis=1)
    winYsrankdf.columns = times.strftime("%Y-%m-%d %H:%M")
    cols = winXmrankdf.columns
    topNcom = []
    for col in cols:
        topNboth = topNNcompind(winXmrankdf[col], winYmrankdf[col], N1,N2)
        topNcom.append(topNboth)
    topN = pd.concat(topNcom,axis=1)
    topN.columns = times.strftime("%Y-%m-%d %H:%M")
    cols = topN.columns
    topNshare = []
    for i in range(0,len(cols)-1):
        topNboth = topNcompsym(topN[cols[i+1]], topN[cols[i]]).reset_index(drop=True)
        topNshare.append(topNboth)
    topNcarried = pd.concat(topNshare,axis=1)
    [buytx, wallet, selltx] = topNbuysell(topN, pbtc)
    return topNshare, topNcarried, buytx, wallet, selltx

def topNbuysell(topN,pbtc):
    #goes through the topN, determines which ones are new to the list (to be bought) and which ones have been removed (to be sold)
    times = topN.columns
    symsprices = []
    buytx = []
    selltx = pd.DataFrame()
    wallet = pd.DataFrame()
    for i in range(0,len(times)):
        walletsyms = wallet.index
        thissyms = topN[times[i]].dropna()
        buysyms = thissyms[~thissyms.isin(walletsyms)]
        sellsyms = walletsyms[~walletsyms.isin(thissyms)]
        buyprices = pbtc[buysyms].loc[times[i]]
        sellprices = pbtc[sellsyms].loc[times[i]]
        buydf = pd.DataFrame(buyprices)
        buytx.append(buydf)
        buydf.columns = ['buy_price']
        buydf['buy_time'] = times[i]
        wallet = wallet.append(buydf)
        selldf = pd.DataFrame(sellprices)
        selldf.columns = ['sell_price']
        selldf['sell_time'] = times[i]
        prevbuyprices = wallet[['buy_price','buy_time']].loc[sellsyms]
        selldf = selldf.join(prevbuyprices)
        selldf['profit%'] = (selldf['sell_price'] - selldf['buy_price']).divide(selldf['buy_price'])
        selltx = selltx.append(selldf)
        wallet = wallet.drop(sellsyms)
        currentprices = pbtc[wallet.index].loc[times[i]]
        wallet['current_price'] = currentprices
        wallet['profit%'] = (wallet['current_price'] - wallet['buy_price']).divide(wallet['buy_price'])
    return buytx, wallet, selltx


def topNcompind(dfX,dfY,N):
    if N != 0:
        df1 = dfX.sort_values(ascending=False).dropna()
        df2 = dfY.sort_values(ascending=False).dropna()
        df1 = df1.iloc[-N:]
        df2 = df2.iloc[-N:]
    topNboth1 = df1[df1.index.isin(df2.index)]
    topNboth2 = df2[df2.index.isin(df1.index)]
    topNboth = pd.DataFrame(topNboth1[topNboth1.index.isin(topNboth2.index)].index.tolist())
    if topNboth.empty:
        topNboth = pd.DataFrame({'0':[np.nan]})
    return topNboth

def topNNcompind(dfX,dfY,N1,N2):
    if N2 != 0:
        df1 = dfX.sort_values(ascending=False).dropna()
        df2 = dfY.sort_values(ascending=False).dropna()
        df1 = df1.iloc[-N1:-N2]
        df2 = df2.iloc[-N1:-N2]
    topNboth1 = df1[df1.index.isin(df2.index)]
    topNboth2 = df2[df2.index.isin(df1.index)]
    topNboth = pd.DataFrame(topNboth1[topNboth1.index.isin(topNboth2.index)].index.tolist())
    if topNboth.empty:
        topNboth = pd.DataFrame({'0':[np.nan]})
    return topNboth

def topNNcompindSortstd(dfX,dfY,N1,N2):
    if N2 != 0:
        df1 = dfX.sort_values(ascending=False).dropna()
        df2 = dfY.sort_values(ascending=False).dropna()
        df1 = df1.iloc[-N1:-N2]
        df2 = df2.iloc[-N1:-N2]
    topNboth1 = df1[df1.index.isin(df2.index)]
    topNboth2 = df2[df2.index.isin(df1.index)]
    topNboth = pd.DataFrame(topNboth1[topNboth1.index.isin(topNboth2.index)].index.tolist())
    if topNboth.empty:
        topNboth = pd.DataFrame({'0':[np.nan]})
    return topNboth

def topNcompsym(df1,df2):
    topNboth1 = df1[df1.isin(df2)]
    topNboth2 = df2[df2.isin(df1)]
    topNboth = pd.DataFrame(topNboth1[topNboth1.isin(topNboth2)])
    return topNboth

def rankattime(endtime, winX, hrX, pbtc):
    starttime = endtime - datetime.timedelta(days=winX)
    windata = pbtc[starttime:endtime]
    [mrankdf, srankdf] = rolling_avg(windata, hrX)
    return mrankdf, srankdf

def rolling_avg(pbtc,hrX):
    # perform the rolling window calculations on the price dataframe
    pctchange = pbtc.pct_change()
    dflogreturn = np.log(1 + pctchange)
    dfdiffwinsum = dflogreturn.rolling(hrX*12).sum()
    rankdf = dfdiffwinsum.rank(1, ascending=False)
    mrankdf = rankdf.mean().rename('mean')
    srankdf = rankdf.std().rename('std')
    return mrankdf, srankdf

def runtestold(window, perend, winX, hrX, winY, hrY, column, N1, N2):
    [topNshare, topNcarried, buytx, wallet, selltx] = rollingtopN(window, perend, winX, hrX, winY, hrY, column, N1, N2)
    # print 'total returns in wallet % ', sum(wallet['profit%'].dropna())
    # print 'total returns % ', sum(selltx['profit%'].dropna())
    selltxnona = selltx.dropna()
    buys = selltxnona.groupby(selltxnona.buy_time).count().buy_price
    [avgbuys, maxbuys, minbuys] = [buys.mean(), buys.max(), buys.min()]
    [avgret, maxret, minret] = [selltxnona['profit%'].mean().round(3),selltxnona['profit%'].max().round(3), selltxnona['profit%'].min().round(3)]
    sellbyday = selltxnona.set_index(['buy_time'])
    dailysell = []
    for ind in sellbyday.index.unique():
        daysell = sellbyday.loc[ind]['profit%'].mean()
        dailysell.append(daysell)
    avgdailyret = round(np.mean(dailysell), 3)
    notxprofit = 1
    for i in dailysell:
        notxprofit = notxprofit * (1 + i)

    print 'Avg # Trans', round(avgbuys,2), '| Max # Trans', maxbuys, '| Min # Trans', minbuys
    print 'Period Return (%)', (round(notxprofit,3)-1)*100, '| Avg Return (%) / Day', \
        (avgdailyret)*100, '| Avg Return (%) / Trans', avgret*100, '| Max Return (%) / Trans', \
        (maxret)*100, '| Min Return (%) / Trans', (minret)*100
    return wallet, selltx

def cleanselltx(selltx):
    #stop gap resource to get rid of NA entries from the selltx record - really need to identify and fix the root cause
    selltx = selltx[~selltx['sell_price'].isnull()]
    selltx = selltx[~selltx['buy_price'].isnull()]
    return selltx

def sells(selltx):
    #only works for one shuffle per day
    if len(selltx) > 0:
        selltx = cleanselltx(selltx)
        selltxstart = selltx.buy_time.iloc[0]
        selltxend = selltx.sell_time.iloc[-1]
        sellperiod = (datetime.datetime.strptime(selltxend,"%Y-%m-%d %H:%M") - datetime.datetime.strptime(selltxstart,"%Y-%m-%d %H:%M")).days
        buys = pd.DataFrame({'buy_price':selltx.buy_price,'buy_time':selltx.buy_time,'sym':selltx.index}).set_index('buy_time')
        buys.index = pd.to_datetime(buys.index).date
        sells = pd.DataFrame({'sell_price':selltx.sell_price,'sell_time':selltx.sell_time,'sym':selltx.index}).set_index('sell_time')
        sells.index = pd.to_datetime(sells.index).date
        dailystats = pd.DataFrame(columns = ['numbuys','numsells','RPD','balance','AvgRPT','returns'],index = pd.date_range(start=selltxstart,periods=sellperiod+2).date)
        wallet = pd.DataFrame(columns = ['sym','qty','buy_price'])
        cashbalance = 1.0
        pastbalance = 1.0
        balance = 1.0
        for day, row in dailystats.iterrows():
            try:
                pastbalance = dailystats.loc[day-datetime.timedelta(days=1)].balance
                balance = copy.deepcopy(pastbalance)
                dayssells = sells.loc[[day]]
                dailystats.loc[day].numsells = len(dayssells)
                RPT = []
                for index, row in dayssells.iterrows():
                    ret = wallet[wallet.sym == row.sym].qty.iloc[0]*row.sell_price
                    cashbalance = cashbalance + ret
                    RPT.append((row.sell_price/(wallet[wallet.sym == row.sym].buy_price.iloc[0])-1))
                    wallet = wallet[wallet.sym != row.sym]
                dailystats.loc[day].AvgRPT = np.mean(RPT)
                dailystats.loc[day].returns = RPT
            except:
                dailystats.loc[day].numsells = 0
                dailystats.loc[day].AvgRPT = 0
                dailystats.loc[day].RPD = 0
                dailystats.loc[day].balance = balance
                dailystats.loc[day].returns = [0]
            try:
                daysbuys = buys.loc[[day]]
                numbuys = len(daysbuys)
                dailystats.loc[day].numbuys = numbuys
                btcperbuy = cashbalance/numbuys
                for index, row in daysbuys.iterrows():
                    wallet = wallet.append(pd.DataFrame({'sym':[row.sym],'qty':[btcperbuy/row.buy_price],'buy_price':[row.buy_price]}),ignore_index=True)
                    cashbalance = cashbalance - btcperbuy
                walletbalance = 0
                for index, row in wallet.iterrows():
                    walletbalance = walletbalance + row.buy_price*row.qty
                balance = cashbalance + walletbalance
                dailystats.loc[day].RPD = balance / pastbalance - 1
                dailystats.loc[day].balance = balance
            except:
                dailystats.loc[day].numbuys = 0
                walletbalance = 0
                for index, row in wallet.iterrows():
                    walletbalance = walletbalance + row.buy_price*row.qty
                balance = cashbalance + walletbalance
                dailystats.loc[day].RPD = balance / pastbalance - 1
                dailystats.loc[day].balance = balance
    else:
        dailystats = pd.DataFrame({'numbuys':0,'numsells':0,'RPD':0,'balance':1.0,'AvgRPT':0,'returns':0},index = [0])
    return dailystats

def runtest(window, perend, winX, hrX, winY, hrY, column, N1, N2, t, p):
    [topNshare, topNcarried, buytx, wallet, selltx] = rollingtopN(window, perend, winX, hrX, winY, hrY, column, N1, N2, t)
    dailystats = sells(selltx)
    if dailystats.index[0] != 0:
        #convert the returns to a flat list
        periodreturn = (round(dailystats.balance[-1],3)-1)*100
        returns = dailystats.returns.copy().dropna().values.tolist()
        returns = list(itertools.chain.from_iterable(returns))
        wins = float(sum(i > 0 for i in returns))
        losses = float(sum(i < 0 for i in returns))
        winloss = wins+losses
        if winloss != 0:
            winprob = wins/(wins+losses)
        else:
            winprob = 0
        AvgTrans = round(np.mean([dailystats.numbuys[:-1], dailystats.numsells[1:]]), 2)
        AvgRPT = round(np.mean(returns[1:]),2)
        AvgRPD = np.mean(dailystats.RPD[1:]) * 100
        StdRPD = np.std(dailystats.RPD[1:]) * 100
        #Sharpe Ratio calculated on the daily returns
        SR = round(np.sqrt(31)*AvgRPD/StdRPD,4)
        try:
            KC = round(winprob-(1-winprob)/(wins/losses),4)
        except:
            KC = 0
        results = [SR, KC, round(periodreturn,3), AvgTrans, round(AvgRPD,2)]
        if p:
            print 'Avg # Trans', AvgTrans, '| Max # Buys', max(dailystats.numbuys[:-1]), '| Min # Buys',min(dailystats.numbuys[:-1])
            print 'Period Return (%)', periodreturn, '| Avg Return (%) / Day', round(AvgRPD,2), \
                '| Avg Return (%) / Trans', AvgRPT
            print 'Kelly Crit ', KC, '| Sharpe Ratio ', SR
    else:
        print 'No Transactions'
        results = [0,0,0,0,0]
    return wallet, selltx, dailystats, results

def runone(win,pend,wX,hX,wY,hY,n1,n2,t,qout,filename):
    print wX, hX, wY, hY, n1, n2
    [wallet, selltx, dailystats, results] = runtest(window=win, perend=pend, winX=wX, hrX=hX,
                                                    winY=wY, hrY=hY, column='price_btc', N1=n1,
                                                    N2=n2, t=t, p=False)
    [SR, KC, periodreturn, AvgTrans, AvgRPD] = [results[0], results[1], results[2], results[3], results[4]]
    RF = pd.DataFrame(
        {'AvgTrans': AvgTrans, 'window': win, 'perend': pend, 'winX': wX, 'hrX': hX, 'winY': wY, 'hrY': hY, 'N1': n1,
         'N2': n2, 'period%return': periodreturn, 'SR': SR, 'KC': KC, 'AvgRPD': AvgRPD}, index=[i],
        columns=['window', 'perend', 'winX', 'hrX', 'winY', 'hrY', 'N1', 'N2', 'period%return', 'SR', 'KC', 'AvgRPD',
                 'AvgTrans'])
    qout.put(RF)
    with open(filename, 'a') as f:
        RF.to_csv(f, header=False)
    return


window = 20
perend = 0
winX = 4
hrX = 4
winY = 2
hrY = 4
N1 = 13
N2 = 1
column = 'price_btc'
t = datetime.datetime(year = 1,month = 1,day=1, hour=1, minute = 0)
p = True

#runtest(window, perend, winX, hrX, winY, hrY, column, N1, N2)
[wallet, selltx, dailystats, results] = runtest(window=window, perend=perend, winX=winX, hrX=hrX, winY=winY, hrY=hrY, column='price_btc', N1=N1, N2=N2, t=t, p=True)

#[wallet, selltx] = runtestold(window=12, perend=1, winX=4, hrX=18, winY=1, hrY=4, column='price_btc', N1=9, N2=1)
sum(selltx['profit%'][(selltx['profit%']>0).values])
sum(selltx['profit%'][(selltx['profit%']<0).values])

##----- Variable Script ---
column = 'price_btc'
times = [datetime.datetime(year = 1,month = 1,day=1, hour=1, minute = 0)]
window = [20]
perend = [0]
winX = [4,6,8,10,12]
hrX = [4,8,12,16,24]
winY = [2,3]
hrY = [4,8,12,16,24]
N1 = [5,8,11,15,20]
N2 = [1,2]

#now = datetime.datetime.now()
#filename = 'ResultFrame{}.csv'.format(now)
ResultFrame = pd.DataFrame(columns = ['window','perend','winX','hrX','winY','hrY','N1','N2','period%return','SR','KC','AvgRPD'])
#ResultFrame.to_csv(filename)

i=0
for win in window:
    for pend in perend:
        for wX in winX:
            for hX in hrX:
                for wY in winY:
                    for hY in hrY:
                        for n1 in N1:
                            for n2 in N2:
                                for t in times:
                                    i=i+1
                                    print wX, hX, wY, hY, n1, n2
                                    [wallet, selltx, dailystats, results] = runtest(window=win, perend=pend, winX=wX, hrX=hX,
                                                                                winY=wY, hrY=hY, column='price_btc', N1=n1,
                                                                                N2=n2, t=t, p=False)

                                    [SR, KC, periodreturn, AvgTrans, AvgRPD] = [results[0],results[1],results[2],results[3],results[4]]
                                    newrow = [win, pend, wX, hX, wY, hY, n1, n2, periodreturn, SR, KC, AvgRPD]
                                    RF = pd.DataFrame({'AvgTrans':AvgTrans,'window':win,'perend':pend,'winX':wX,'hrX':hX,'winY':wY,'hrY':hY,'N1':n1,'N2':n2,'period%return':periodreturn,'SR':SR,'KC':KC,'AvgRPD':AvgRPD},index=[i],columns = ['window','perend','winX','hrX','winY','hrY','N1','N2','period%return','SR','KC','AvgRPD','AvgTrans'])
                                    with open('mycsv.csv', 'a') as f:
                                        RF.to_csv(f,header=False)
                                    ResultFrame = ResultFrame.append(RF)[RF.columns.tolist()]

results = pd.read_csv('mycsv.csv',index_col=0, names = ['window','perend','winX','hrX','winY','hrY','N1','N2','period%return','SR','KC','AvgRPD','AvgTrans'])
nas = results[results['period%return'].isnull()]


filename = 'ResultFrame{}.csv'.format(now)
ResultFrame = pd.DataFrame(columns = ['window','perend','winX','hrX','winY','hrY','N1','N2','period%return','SR','KC','AvgRPD','AvgTrans'])
ResultFrame.to_csv(filename)
