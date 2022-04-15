import cugraph
import cudf
from cugraph.experimental import PropertyGraph
import dgl.dataloading.graphstorage as graphstorage


def read_cugraph(graph_path, feat_path, self_loop=False):
    # graph path '/home/xiaoyunw/cugraph/datasets/cora/cora.cites'
    # feat_path '/home/xiaoyunw/cugraph/datasets/cora/cora.content'
    cora_M = cudf.read_csv(graph_path, sep = '\t', header = None)
    cora_content = cudf.read_csv(feat_path, sep = '\t', header = None)
    # the last column is true label
    cora_content1 = cora_content.drop (columns = '1434')
    # add weight into graph
    cora_M['weight'] = 1.0

    # add features to nodes and edges
    pg = PropertyGraph()

    pg.add_edge_data(cora_M, vertex_col_names=("0","1"))
    pg.add_vertex_data(cora_content1, vertex_col_name = "0")

    pg._vertex_prop_dataframe.drop(columns = ['0'], inplace = True)
    pg._edge_prop_dataframe.drop(columns = ['0', '1'], inplace = True)

    labels = cora_content['1434']
    # create DGL graph storage
    #gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore = dgl.dataloading.graphstorage(pg)

    return gstore, labels



if __name__ == '__main__':
    graph_path = '/home/xiaoyunw/cugraph/datasets/cora/cora.cites'
    feat_path = '/home/xiaoyunw/cugraph/datasets/cora/cora.content'
    gstore, labels = read_cugraph(graph_path, feat_path)
    print (gstore)
    print (labels)


