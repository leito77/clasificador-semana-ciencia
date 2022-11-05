import numpy as np
import pandas as pd

def get_data_out_of_batches(model, dataset, FLAGS):
    num_batches = int(dataset[0].shape[0]/FLAGS.batch_size)    
    A = []
    X = []
    P = []
    labels_batch_list = []
    for _iter in range(num_batches):
        A_batch = dataset[0][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
        X_batch = dataset[1][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
        P_batch = dataset[2][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
        A.append(A_batch)
        X.append(X_batch)
        P.append(P_batch)
    return (np.asarray(A), np.asarray(X), np.asarray(P))

def build_df(nodes, labels):
    # remove all zero rows 
    df_nodes = pd.DataFrame(np.array(nodes))
    nodes_reduced = df_nodes.loc[~(df_nodes==0).all(axis=1)]

    # create labels dataframe 
    df_labels = pd.DataFrame(np.array(labels), columns=['activity'])
  
    # left join labels by index of nodes_reduced 
    df_nodes_labels = pd.merge(nodes_reduced, df_labels, left_index=True, right_index=True)
    return df_nodes_labels
