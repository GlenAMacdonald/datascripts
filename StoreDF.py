import matplotlib
import pandas as pd
import datetime
import coinmarketcap as cmc
sys.path.extend(['/home/Spare/CC/datascripts'])
import getdata

datasetname = 'cmcdataset'
market = cmc.Market()

def select_HDFstore(datasetname):
    store = pd.HDFStore('/home/Spare/CC/data/{}.h5'.format(datasetname))
    return store

def store_dset(datasetname, df, key):
    store = select_HDFstore(datasetname)
    store.put(key, df, format='table')
    store.close()
    return

def append_dset(datasetname, df, key):
    store = select_HDFstore(datasetname)
    store.put(key, df, format='table',append=True)
    store.close()
    return

def identify_top200():
    df1 = pd.DataFrame(market.ticker('?start=0&limit=99'))
    df2 = pd.DataFrame(market.ticker('?start=100&limit=199'))
    top200 = pd.concat([df1,df2],ignore_index=True)
    return top200

def init_dl_top200(top200):
    syms = top200.symbol.tolist()
    tablelist = getdata.get_table_list(getdata.bqc,getdata.job_config)
    data = getdata.get_many_syms(syms, tablelist, getdata.bqc, getdata.job_config, 'all')
    data.columns = data.columns.astype(str)
    data.index.set_levels(data.index.levels[0].astype(str), level=0, inplace=True)
    return data

def init_dset(datasetname, data):
    store = select_HDFstore(datasetname)
    syms = data.index.levels[0]
    for sym in syms:
        store.put(sym,data.loc[sym],format='table')
    tlisth5 = pd.DataFrame(syms, columns=['sym'])
    tlisth5['last_updated'] = 0
    tlisth5.set_index('sym', drop=True, inplace=True)
    tlisth5.index.name = None
    for sym in syms:
        tlisth5.loc[sym] = data.loc[sym].index[-1].strftime("%Y-%m-%d %H:%M")
    store.put('tablelistH5', tlisth5, format='table')
    store.close()
    return

def ud_dset_tlist(datasetname):
    store = select_HDFstore(datasetname)
    tlisth5 = store.get('tablelistH5')
    syms = tlisth5.index
    tlisth5 = pd.DataFrame(syms, columns=['sym'])
    tlisth5['last_updated'] = 0
    tlisth5.set_index('sym', drop=True, inplace=True)
    tlisth5.index.name = None
    for sym in syms:
        data = store.get(sym)
        tlisth5.loc[sym] = data.index[-1].strftime("%Y-%m-%d %H:%M")
    store.put('tablelistH5',tlisth5,format='table')
    store.close()
    return

def get_last_update(datasetname):
    store = select_HDFstore(datasetname)
    tlisth5 = store.get('tablelistH5')
    store.close()
    return tlisth5

def update_top200(datasetname):
    store = select_HDFstore(datasetname)
    tlisth5 = store.get('tablelistH5')
    top200 = identify_top200()
    top200syms = top200.symbol.tolist()
    tlisth5syms = tlisth5.index.tolist()
    newsyms = list(set(top200syms).difference(tlisth5syms))
    existingsyms = list(set(top200syms).intersection(tlisth5syms))
    tablelist = getdata.get_table_list(getdata.bqc, getdata.job_config)
    lastupdated = []
    if newsyms:
        for sym in newsyms:
            newdf = getdata.get_sym(sym, tablelist, getdata.bqc, getdata.job_config)
            store.put(sym,newdf,format='table')
            newrow = pd.DataFrame(0,columns=['last_updated'],index=[sym])
            tlisth5 = tlisth5.append(newrow)
        store.put('tablelistH5',tlisth5,format='table')
    for sym in existingsyms:
        lastupdated.append(tlisth5.loc[sym][0])
    multidf = getdata.upd_many_syms(existingsyms, tablelist, lastupdated, getdata.bqc, getdata.job_config)
    for sym in multidf.index.levels[0]:
        df = multidf.loc[sym]
        store.append(sym,df,format='table')
    ud_dset_tlist(datasetname)
    store.close()
    return

def restore_tlisth5(datasetname):
    store = select_HDFstore(datasetname)
    keys = store.keys()
    stripped_keys = list(map(lambda x: x.strip('/'), keys))
    stripped_keys.remove('tablelistH5')
    tlisth5 = pd.DataFrame(stripped_keys, columns=['sym'])
    tlisth5['last_updated'] = 0
    tlisth5.set_index('sym', drop=True, inplace=True)
    tlisth5.index.name = None
    for sym in stripped_keys:
        data = store.get(sym)
        tlisth5.loc[sym] = data.index[-1].strftime("%Y-%m-%d %H:%M")
    store.put('tablelistH5',tlisth5,format='table')
    store.close()
    return

top200 = identify_top200()
lastupdate = get_last_update(datasetname)
update_top200(datasetname)
