import matplotlib
from PyEMD import EEMD
import numpy as np
import pylab as plt
import sys
sys.path.extend(['/home/Spare/CC/datascripts'])
import StoreDF
import mungeData
from progress.bar import Bar
from tqdm import tqdm

datasetname = 'cmcdataset'

def get_data_for_EEMD(sym,column,per,perend):
    datasetname = 'cmcdataset'
    store = StoreDF.select_HDFstore(datasetname)
    [window, period, periodend] = mungeData.conv_win_2_block(0, per, perend)
    df = store.get(sym)
    df = df.loc[~df.index.duplicated(keep='first')]
    try:
        datablock = df[column].iloc[-(period + periodend):-periodend]
    except Exception as e:
        print 'Period outside of length of block ' , e
        datablock = np.nan()
    return datablock

def rolling_EEMD(sym,column,win,per,perend):
    datablock = get_data_for_EEMD(sym,column,per,perend)
    winlength = int(win * 24 * 12)
    t = np.linspace(0, winlength-1, winlength)
    eemd = EEMD()
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    nIMF = []
    last = []
    dlast = []
    timelist = []
    timearray = datablock.index[(winlength):]
    for i in tqdm(range(winlength,datablock.__len__(),1)):
        datawindow = datablock.iloc[i-winlength:i]
        # Execute EEMD on window
        maximf = 6
        eIMFs = eemd.eemd(datawindow.tolist(), t, max_imf=maximf)
        nIMFs = eIMFs.shape[0]
        nIMF.append(nIMFs)
        last.append(eIMFs[0:nIMFs,-1])
        d = eIMFs[0:nIMFs,-1] - eIMFs[0:nIMFs,-2]
        dlast.append(d)
        timelist.append(datawindow.index[-1])
    lastdf = pd.DataFrame(last,index = timearray)
    dlastdf = pd.DataFrame(dlast,index = timearray)
    return [lastdf,dlastdf]


sym = 'BTC'
column = 'price_usd'
win = 0.1
per = 1.0
perend = 0
datablock = get_data_for_EEMD(sym,column,per,perend)

[lastdf, dlastdf] = rolling_EEMD(sym,column,win,per,perend)

lastdf = pd.read_csv('lastdf.csv')
dlastdf = pd.read_csv('dlastdf.csv')
lastdf.drop('timestamp',axis=1, inplace = True)
dlastdf.drop('timestamp',axis=1, inplace = True)
lastdf.fillna(0, inplace=True)
dlastdf.fillna(0, inplace=True)

winlength = int(win * 24 * 12)
data = datablock.iloc[winlength:]
data.reset_index(inplace = True, drop=True)
trend = lastdf[2:].sum(axis=1)
noise = lastdf[:1].sum(axis=1)

dtrend = dlastdf.iloc[:,2:].sum(axis=1)
dnoise = dlastdf.iloc[:,2:].sum(axis=1)

dataup = (data*(dtrend > 0)).replace(0, np.nan)
datadn = (data*(dtrend < 0)).replace(0, np.nan)
datan = (data*(dtrend == 0)).replace(0, np.nan)

tp = np.linspace(0, datablock.__len__()-winlength-1, datablock.__len__()-winlength)
tp = dtrend.index
plt.figure(figsize=(12,9))
#plt.sub plot(2, 1, 1)
plt.plot(tp,data,'k')
plt.plot(tp,dataup,'g')
plt.plot(tp,datadn,'r')

plt.subplot(2, 1, 2)
plt.plot(tp,dtrend,'b')
plt.axhline(y=0,color = 'r', linestyle='-')





