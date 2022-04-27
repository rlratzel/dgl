"""
cugraph sampling test on Benchmark Datasets
"""
import numpy as np
import pandas as pd
#  Import the modules
import cugraph
import cudf
import dgl
# system and other
import gc
import os
import time
import random
import cupy
# MTX file reader
from scipy.io import mmread
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch
# dlpack can only used for pytorch > 1.10 and cupy > 10

def read_and_create(datafile):
    # print('Reading ' + str(datafile) + '...')
    M = mmread(datafile).asfptype()

    _gdf = cudf.DataFrame()
    _gdf['src'] = M.row
    _gdf['dst'] = M.col
    _gdf['wt'] = 1.0

    _g = cugraph.Graph()
    _g.from_cudf_edgelist(_gdf, source='src', destination='dst', edge_attr='wt', renumber=False)

    # print("\t{:,} nodes, {:,} edges".format(_g.number_of_nodes(), _g.number_of_edges() ))

    return _g

def cugraphSampler(g, nodes, fanouts, edge_dir='in', prob=None, replace=False,
                     copy_ndata=True, copy_edata=True, _dist_training=False, exclude_edges=None):
    # from here get in a new for loop
    # ego_net return edge list
    num_nodes = len(nodes)
    #current_seeds = nodes.reindex(index = np.arange(0, num_nodes))
    current_seeds = cupy.asarray(nodes)
    current_seeds = cudf.Series(current_seeds)
    blocks = []
    #seeds = cudf.Series(nodes.to_array())

    for fanout in fanouts:
        ego_edge_list, seeds_offsets = cugraph.community.egonet.batched_ego_graphs(g, current_seeds, radius = 1)
        #print ("current_seeds", current_seeds)
        print ("fanout", fanout)
        #all_parents = cupy.ndarray(fanout*len(current_seeds))
        #all_children = cupy.ndarray(fanout*len(current_seeds))
        all_parents = cupy.ndarray(0)
        all_children = cupy.ndarray(0)
        #print ("all parents", all_parents)
    # filter and get a certain size neighborhood
        for i in range(1, len(seeds_offsets)):
            pos0 = seeds_offsets.values_host[i-1]
            pos1 = seeds_offsets.values_host[i]
            edge_list = ego_edge_list[pos0:pos1]
            # get randomness fanout
            filtered_list = edge_list[edge_list ['dst']== current_seeds[i-1]]
             
            # get sampled_list
            if len(filtered_list) > fanout:
                sampled_indices = random.sample(filtered_list.index.to_arrow().to_pylist(), fanout)
                filtered_list = filtered_list.reindex(index = sampled_indices)
                
            children = cupy.asarray(filtered_list['src'])
            parents = cupy.asarray(filtered_list['dst'])
            # copy the src and dst to cupy array
            all_parents = cupy.append(all_parents, parents)
            all_children = cupy.append(all_children, children)
            #print (len(test_parents)) 

        # generate dgl.graph and  blocks
        sampled_graph = dgl.graph ((all_children,all_parents))
        #print(all_parents)
        #print(all_children)
        #print(sampled_graph.edges())
        #print(seeds.to_array())
        #eid = sampled_graph.edata[dgl.EID]
        #block =dgl.to_block(sampled_graph,current_seeds.to_array())
        #block.edata[dgl.EID] = eid
        #current_seeds = block.srcdata[dgl.NID]
        #current_seeds = cudf.Series(current_seeds.cpu().detach().numpy())

        #blocks.insert(0, block)
        # end of for

    return sampled_graph


if __name__ == '__main__':
    data = ['polbooks']#, 'as-Skitter', 'citationCiteseer', 'caidaRouterLevel', 'coAuthorsDBLP', 'coPapersDBLP']
    for file_name in data:
        G_cu = read_and_create('/home/xiaoyunw/cugraph/datasets/'+ file_name + '.mtx') 
        nodes = G_cu.nodes()#.to_array().tolist()
        #print(nodes.index)
        num_nodes = G_cu.number_of_nodes()
        #num_seeds_ = [1000, 3000, 5000, 10000]
        # just test 1 epoch
        batch_size = 10
        num_batch = num_nodes/batch_size
        print (num_batch)
        # in each epoch shuffle the nodes
        shuffled_nodes = np.arange(num_nodes)
        #print(len(nodes), len(shuffled_nodes))
        np.random.shuffle(shuffled_nodes)
        print(type(nodes))
        #shuffled_nodes = cudf.Series(shuffled_nodes)
        #new_nodes = nodes.reindex(index = shuffled_nodes)
        shuffled_nodes = torch.tensor(shuffled_nodes) 
        print (nodes)
        for i in range(int(num_batch)-1):
            blocks = cugraphSampler(G_cu, shuffled_nodes[i*batch_size: (i+1)*batch_size], [5,10]) 


           
