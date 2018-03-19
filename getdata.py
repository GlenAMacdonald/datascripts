from google.cloud import bigquery
import pandas as pd
import Queue
import logging


# Set project Variables
projectid = "pythonproject-190701"
dataset_id = 'cmcdataset'
bqc = bigquery.Client(project=projectid)
dataset_ref = bqc.dataset(dataset_id)


def query(query,bqc):
    query_job = bqc.query(query)
    return query_job

def query_job_2df(query_job):
    # Convert the Results Query to a List of lists
    data = list(x for x in query_job.result())
    # Convert the list of lists to a dataframe with columns based on the fields listed in the first row.
    df = pd.DataFrame.from_records(data[:],columns = sorted(data[0]._xxx_field_to_index, key=data[0]._xxx_field_to_index.__getitem__))
    return df

q = 'SELECT * FROM cmcdataset.BTC ORDER BY timestamp desc'
query_job = query(q,bqc)
df = query_job_2df(query_job)

