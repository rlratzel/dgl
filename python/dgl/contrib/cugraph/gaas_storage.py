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

# NOTE: Requires cuGraph nightly cugraph-22.06.00a220417 or later

import random
import time
import torch

import dgl


class TorchTensorGaasGraphDataProxy:
    """
    Implements a partial Torch Tensor interface that forwards requests to a
    GaaS server maintaining the actual data in a graph instance.

    The interface supported consists of only the APIs specific DGL workflows
    need - anything else will raise AttributeError.
    """
    _data_categories = ["vertex", "edge"]

    def __init__(self, gaas_client, gaas_graph_id, data_category):
        if data_category not in self._data_categories:
            raise ValueError("data_category must be one of "
                             f"{self._data_categories}, got {data_category}")
        self.__client = gaas_client
        self.__graph_id = gaas_graph_id
        self.__category = data_category

    def __getitem__(self, index):
        """
        Returns a torch.Tensor containing the edge or vertex data (based on the
        instance's data_category) for index, retrieved from graph data on the
        instance's GaaS server.
        """
        # tensor is a transposed dataframe (tensor[0] is df.iloc[0])
        if isinstance(index, torch.Tensor):
            index = [int(i) for i in index]

        if self.__category == "edge":
            data = self.__client.get_graph_edge_dataframe_rows(
                index_or_indices=index, graph_id=self.__graph_id)
        else:
            data = self.__client.get_graph_vertex_dataframe_rows(
                index_or_indices=index, graph_id=self.__graph_id)

        torch_data = torch.from_numpy(data)
        return torch_data.to(torch.float32)

    @property
    def shape(self):
        if self.__category == "edge":
            return self.__client.get_graph_edge_dataframe_shape(
                graph_id=self.__graph_id)
        else:
            return self.__client.get_graph_vertex_dataframe_shape(
                graph_id=self.__graph_id)



class GaasGraphStorage:
    """
    Duck-typed version of the DGL GraphStorage class made for GaaS
    """
    def __init__(self, gaas_client, gaas_graph_id):
        # FIXME: input type check?
        self.__client = gaas_client
        self.__graph_id = gaas_graph_id
        self.__extracted_subgraph_id = None

    @property
    def ndata(self):
        return TorchTensorGaasGraphDataProxy(
            self.__client, self.__graph_id, "vertex")

        # self._ndata is the entire PG._vertex_prop_dataframe
        # the return value is a torch.Tensor
        # the return value is used like this:
        #      batch_inputs = nfeat[input_nodes].to(device)
        #ndata_capsule = self._ndata.to_dlpack()
        #nfeat = torch.from_dlpack(ndata_capsule)
        #return nfeat.to(torch.float32)

    @property
    def edata(self):
        return TorchTensorGaasGraphDataProxy(
            self.__client, self.__graph_id, "vertex")
        #return self._edata

    def get_node_storage(self, key, ntype=None):
        raise NotImplementedError

        node_col = self.graphstore.get_node_storage(key, ntype)
        import cupy
        return torch.as_tensor(cupy.asarray(node_col))

    def get_edge_storage(self, key, etype=None):
        raise NotImplementedError
        import cupy
        edge_col = self.graphstore.get_edge_storage(key, etype)
        return torch.as_tensor(cupy.asarray(edge_col))

    # Required for checking whether single dict is allowed for ndata and edata
    @property
    def ntypes(self):
        raise NotImplementedError
        from cugraph.experimental import PropertyGraph
        data_ntypes = self._ndata[PropertyGraph.type_col_name]
        # TODO: double check the return type
        return data_ntypes

    @property
    def canonical_etypes(self):
        raise NotImplementedError("canonical not implemented")

    def etypes(self):
        raise NotImplementedError
        from cugraph.experimental import PropertyGraph
        data_etypes = self._edata[PropertyGraph.type_col_name]
        return data_etypes

    def sample_neighbors(self, seed_nodes, fanout, edge_dir='in', prob=None,
                         exclude_edges=None, replace=False,
                         output_device=None):
        """
        Return a DGLGraph which is a subgraph induced by sampling neighboring
        edges ofthe given nodes.
        See ``dgl.sampling.sample_neighbors`` for detailed semantics.
        Parameters
        ----------
        seed_nodes : Tensor or dict[str, Tensor]
            Node IDs to sample neighbors from.
            This argument can take a single ID tensor or a dictionary of node
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        fanout : int or dict[etype, int]
            The number of edges to be sampled for each node on each edge type.
            This argument can take a single int or a dictionary of edge types
            and ints. If a single int is given, DGL will sample this number of
            edges for each node for every edge type.
            If -1 is given for a single edge type, all the neighboring edges
            with that edge type will be selected.
        prob : str, optional
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node.  The feature must have only
            one element for each edge.
            The features must be non-negative floats, and the sum of the
            features of inbound/outbound edges for every node must be positive
            (though they don't have to sum up to one).  Otherwise, the result
            will be undefined. If :attr:`prob` is not None, GPU sampling is
            not supported.
        exclude_edges: tensor or dict
            Edge IDs to exclude during sampling neighbors for the seed nodes.
            This argument can take a single ID tensor or a dictionary of edge
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        replace : bool, optional
            If True, sample with replacement.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            A sampled subgraph with the same nodes as the original graph, but
            only the sampled neighboring edges.  The induced edge IDs will be
            in ``edata[dgl.EID]``.
        """
        print("----> creating sampled graph from GaasGraphStorage.sample_neighbors()...", flush=True)
        if torch.is_tensor(seed_nodes):
            seed_nodes = seed_nodes.tolist()

        # self.__graph_id is the ID to the PropertyGraph on the GaaS server.
        # Extract a subgraph (in this case, the entire graph) - if one has not
        # been extracted before - as a cugraph.Graph on the server in order to
        # run algos (ie. sampling) on it.
        if self.__extracted_subgraph_id is None:
            self.__extracted_subgraph_id = \
                self.__client.extract_subgraph(allow_multi_edges=True,
                                               graph_id=self.__graph_id)

        st=time.time()
        print(f"  calling egonet on {len(seed_nodes)} seed_nodes...", flush=True)
        (srcs, dsts, weights, seeds_offsets) = \
            self.__client.batched_ego_graphs(
                seed_nodes, radius=1, graph_id=self.__extracted_subgraph_id)
        print(f"   back from egonet, time was {time.time()-st}...", flush=True)

        parents_nodes = []
        children_nodes = []
        for i in range(1, len(seeds_offsets)):
            pos0 = seeds_offsets[i-1]
            pos1 = seeds_offsets[i]
            seed_to_match = seed_nodes[i-1]
            # filter the list of edges by picking only those edges where the dst
            # vertex is the starting seed for the sample
            filtered_indices = [i for i in range(pos0, pos1)
                                if dsts[i] == seed_to_match]

            if len(filtered_indices) > fanout:
                filtered_indices = random.sample(filtered_indices, fanout)

            # FIXME: why are children_nodes src and parents_nodes dst here?
            children_nodes += [srcs[i] for i in filtered_indices]
            parents_nodes += [dsts[i] for i in filtered_indices]

        num_edges = len(children_nodes)
        st=time.time()
        print("   getting edge IDs...", flush=True)
        edge_ID_list = self.__client.get_edge_IDs_for_vertices(
            parents_nodes, children_nodes, self.__extracted_subgraph_id)
        print(f"   done getting edge IDs, time was {time.time()-st}...", flush=True)

        # construct dgl graph, want to double check if children and parents
        # are in the correct order
        sampled_graph = dgl.graph((children_nodes, parents_nodes))
        # add '_ID'
        num_edges = len(children_nodes)
        sampled_graph.edata['_ID'] = torch.tensor(edge_ID_list)
        # to device function move the dgl graph to desired devices
        if output_device is not None:
            sampled_graph.to_device(output_device)

        print("returning sampled graph from GaasGraphStorage.sample_neighbors()\n", flush=True)
        return sampled_graph

    # Required in Cluster-GCN
    def subgraph(self, nodes, relabel_nodes=False, output_device=None):
        """Return a subgraph induced on given nodes.
        This has the same semantics as ``dgl.node_subgraph``.
        Parameters
        ----------
        nodes : nodes or dict[str, nodes]
            The nodes to form the subgraph. The allowed nodes formats are:
            * Int Tensor: Each element is a node ID. The tensor must have the
             same device type and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether node :math:`i` is in the subgraph.
             If the graph is homogeneous, directly pass the above formats.
             Otherwise, the argument must be a dictionary with keys being
             node types and values being the node IDs in the above formats.
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError
        """
        sampled_cugraph = self.graphstore.node_subgraph(nodes)
        # the return type is cugraph subgraph
        sample_graph = cugraphToDGL(sampled_cugraph)
        sample_graph.to_device(output_device)
        return sample_graph
        """

    # Required in Link Prediction
    # relabel = F we use dgl functions,
    # relabel = T, we need delete codes and relabel
    def edge_subgraph(self, edges, relabel_nodes=False, output_device=None):
        """
        Return a subgraph induced on given edges.
        This has the same semantics as ``dgl.edge_subgraph``.
        Parameters
        ----------
        edges : edges or dict[(str, str, str), edges]
            The edges to form the subgraph. The allowed edges formats are:
            * Int Tensor: Each element is an edge ID. The tensor must have the
              same device type and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether edge :math:`i` is in the subgraph.
            If the graph is homogeneous, one can directly pass the above
            formats. Otherwise, the argument must be a dictionary with keys
            being edge types and values being the edge IDs in the above formats
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError

    # Required in Link Prediction negative sampler
    def find_edges(self, edges, etype=None, output_device=None):
        """Return the source and destination node IDs given the edge IDs within
         the given edge type.
         return type is tensor need to change.
        """
        raise NotImplementedError
        """
        # edges are a range of edge IDs, for example 0-100
        selected_edges = (
            self._edata[self._edata['_TYPE_'] == etype].iloc[edges])
        src_nodes = selected_edges['_SRC_']
        dst_nodes = selected_edges['_DST_']
        src_nodes_tensor = torch.as_tensor(src_nodes, device=output_device)
        dst_nodes_tensor = torch.as_tensor(dst_nodes, device=output_device)
        return src_nodes_tensor, dst_nodes_tensor
        """

    # Required in Link Prediction negative sampler
    def num_nodes(self, ntype):
        """Return the number of nodes for the given node type."""
        raise NotImplementedError
        """
        # use graphstore function
        return self._ndata[self._ndata['_TYPE_'] == ntype].shape[0]
        """

    def global_uniform_negative_sampling(self, num_samples,
                                         exclude_self_loops=True,
                                         replace=False, etype=None):
        """
        Per source negative sampling as in ``dgl.dataloading.GlobalUniform``
        """
        raise NotImplementedError("canonical not implemented")
