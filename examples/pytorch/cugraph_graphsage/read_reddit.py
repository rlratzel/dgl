# Copyright (c) 2019-2022, NVIDIA CORPORATION.   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.   
# You may obtain a copy of the License at   #   
# http://www.apache.org/licenses/LICENSE-2.0   #   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS,   
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   
# See the License for the specific language governing permissions and   
# limitations under the License.

import cugraph
import cudf
from cugraph.experimental import PropertyGraph
import dgl.contrib.cugraph.graph_storage as graphstorage
import numpy as np
import random
import sklearn
import pandas as pd
import scipy as sp

def read_reddit(raw_path, self_loop=False):
    #url = 'https://data.dgl.ai/dataset/reddit.zip'
    #raw_path = "/home/xiaoyunw/Downloads/reddit"
    coo_adj = sp.load_npz(os.path.join(raw_path, "reddit_graph.npz"))
    csr_adj = coo_adj.tocsr()
    offsets = pd.Series(csr_adj.indptr)
    indices = pd.Series(csr_adj.indices)
    graph = cugraph.from_adjlist(offsets, indices, None)

    # features and labels
    reddit_data = np.load(os.path.join(raw_path, "reddit_data.npz"))
    features = reddit_data["feature"]
    labels = reddit_data["label"]
    # tarin/val/test indices
    node_types = reddit_data["node_types"]
    train_mask = (node_types == 1)
    val_mask = (node_types == 2)
    test_mask = (node_types == 3)
    # add features to nodes and edges
    pg = PropertyGraph()

    pg.add_edge_data(graph, vertex_col_names=("0","1"))

    pg.add_vertex_data(reddit_data, vertex_col_name = "0")

    gstore = dgl.contrib.cugraph.CuGraphStorage(pg)

    return gstore, labels, train_mask, val_mask, test_mask



#if __name__ == '__main__':
    

