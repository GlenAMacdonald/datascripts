import pandas as pd

datasetname = 'cmcdataset'

def select_HDFstore(datasetname):
    store = pd.HDFStore('/home/Spare/CC/data/{}.h5'.format(datasetname))
    return store

# Note: store['dset'] = df  copies the dataframe into the store.

def get_dset(datasetname):
    dset = pd.read_hdf('/home/Spare/CC/data/{}.h5'.format(datasetname))
    return dset


