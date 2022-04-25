# Copyright (c) 2021-2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
from cugraph.experimental import PropertyGraph
from dgl.contrib.cugraph import CuGraphStorage


def read_cora(graph_path, feat_path, self_loop=False):
    """
    Read the cora dataset, create and load a Property Graph,
    Then creat a CuGraphStorage object
    """
    pg = PropertyGraph()

    # load the edges and add a weight column
    cora_M = cudf.read_csv(graph_path, sep='\t', header=None)
    cora_M['weight'] = 1.0
    pg.add_edge_data(cora_M, vertex_col_names=("0", "1"))
    pg._edge_prop_dataframe.drop(columns=['0', '1'], inplace=True)

    # now load the wertex data, the last column (idx 1434) is the labels
    cora_content = cudf.read_csv(feat_path, sep='\t', header=None)
    labels = cora_content['1434']
    cora_content.drop(columns='1434', inplace=True)
    pg.add_vertex_data(cora_content, vertex_col_name="0")
    pg._vertex_prop_dataframe.drop(columns=['0'], inplace=True)

    gstore = CuGraphStorage(pg)

    return gstore, labels


if __name__ == '__main__':
    graph_path = './datasets/cora/cora.cites'
    feat_path = './datasets/cora/cora.content'
    gstore, labels = read_cora(graph_path, feat_path)
    print(gstore)
    print(labels)
