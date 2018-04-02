from google.cloud import bigquery
import pandas as pd
import Queue
import threading
import logging


# Set project Variables
projectid = "pythonproject-190701"
dataset_id = 'cmcdataset'
bqc = bigquery.Client(project=projectid)
job_config = bigquery.QueryJobConfig()
job_config.use_legacy_sql = False
dataset_ref = bqc.dataset(dataset_id)


def query(query,bqc,job_config):
    query_job = bqc.query(query,job_config=job_config)
    return query_job

def query_job_2df(query_job):
    # Convert the Results Query to a List of lists
    data = list(x for x in query_job.result())
    # Convert the list of lists to a dataframe with columns based on the fields listed in the first row.
    df = pd.DataFrame.from_records(data[:],columns = sorted(data[0]._xxx_field_to_index, key=data[0]._xxx_field_to_index.__getitem__))
    return df

def get_table_list(bqc,job_config):
    q = "SELECT * FROM `cmcdataset.Table_List` ORDER BY TableID"
    query_job = query(q,bqc,job_config)
    df = query_job_2df(query_job)
    #does not currently support currencies that have the same symbol
    df.drop_duplicates('symbol', inplace=True)
    df.set_index('symbol',inplace = True)
    return df

def get_sym(sym,tablelist,bqc,job_config):
    tableid = tablelist.loc[sym]['TableID']
    q = "SELECT * from `cmcdataset.{}` ORDER BY timestamp".format(tableid)
    query_job = query(q,bqc,job_config)
    df = query_job_2df(query_job)
    df.set_index('timestamp',inplace=True)
    return df

def get_symq(symq,qout,tablelist,bqc,job_config):
    sym = symq.get()
    df = get_sym(sym,tablelist,bqc,job_config)
    qout.put([df,sym])
    symq.task_done()
    return

def get_sym_loop(symq,qout,tablelist,bqc,job_config):
    while symq.qsize() > 0:
        get_symq(symq, qout, tablelist, bqc, job_config)
    return

def get_many_syms(syms,tablelist,bqc,job_config):
    dflist = []
    symq = Queue.Queue()
    qout = Queue.Queue()
    numthreads = 5
    threads = []
    symlist = []
    for sym in syms:
        symq.put((sym))
    for i in range(numthreads):
        t = threading.Thread(target=get_sym_loop,args = (symq,qout,tablelist,bqc,job_config))
        threads.append(t)
    for i in threads:
        i.start()
    for i in threads:
        i.join()
    for i in range(qout.qsize()):
        [df,sym] = qout.get()
        symlist.append(sym)
        dflist.append(df)
    multidf = pd.concat(dflist, keys=symlist)
    return multidf

syms = ['BTC']
tablelist = get_table_list(bqc,job_config)
df = get_many_syms(syms,tablelist,bqc,job_config)
dset = df

#sym = 'ETH'
#df = get_sym(sym,tl,bqc,job_config)
#sym2 = 'BTC'
#df2 = get_sym(sym2,tl,bqc,job_config)

