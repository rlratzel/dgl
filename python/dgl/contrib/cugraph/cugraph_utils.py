# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import cugraph
import dgl
from cugraph.experimental import PropertyGraph


# from cugraph to DGL using 
def toDGL(graph):
    """
    Convert fromn a cuGraph graph to a DGLGraph
    Parameters
    ----------
    graph : cugraph.Graph
        A cuGraph Graph object in GPU memory

    Returns
    -------
    g_dgl : DGLGraph
    """
    edgelist = graph.edges()
    src = cupy.asarray(edgelist['src'])
    dst = cupy.asarray(edgelist['dst'])
    g_dgl = dgl.graph((src, dst))
    return g_dgl
