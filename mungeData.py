import pandas as pd
import numpy

datasetname = 'cmcdataset'

dset = get_dset(datasetname)

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
    dflist = []
    syms = dset.index.levels[0]
    for sym in syms:
        volume = pd.DataFrame(dset.loc[sym].vol_24h)
        vmc = dset.loc[sym].vol_24h/dset.loc[sym].mkt_cap
        vdiff = volume.vol_24h.diff()
        volest = volavg + vdiff
    return

pusd = pd.DataFrame(dset.loc[sym].price_usd)
pusd = pusd.price_usd



