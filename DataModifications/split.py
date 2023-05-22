import os.path as osp

import dgl
import numpy as np

from Utils import io

"""
I have an issue with fitting the turnaround time for lightGCN benchmarks. 
I will reduce the size of the data by .5. 
"""


def create_reduced_graph(original_graph, k=.5):
    """
    I cannot simply reassign the graph size in graph.nodes()
    So I need to create a new graph.
    dgl.graph takes as input:
    dgl.graph(data)
    Where data: The data for constructing a graph, which takes the form of (U,V) (U[i],V[i])
    forms the edge with ID i in the graph.
    """
    num_nodes = original_graph.num_nodes()
    new_num_nodes = int(num_nodes*k)
    new_nodes = set(train_graph.nodes()[:new_num_nodes].numpy())
    new_nodes = dict.fromkeys(new_nodes, True)
    E_src, E_dst = original_graph.edges()
    E_src, E_dst = E_src.numpy(), E_dst.numpy()


    new_E_src, new_E_dst = [], []
    for i in range(len(E_src)):  # Loop should be of time O(7n) == O(n)
        E_src_i, E_dst_i = E_src[i], E_dst[i]
        src_i = new_nodes.get(E_src_i, False)  # Dict look up is O(1)
        dst_i = new_nodes.get(E_dst_i, False)

        if (src_i and dst_i):
            new_E_src.append(E_src_i)
            new_E_dst.append(E_dst_i)

    new_graph = dgl.graph((new_E_src, new_E_dst))
    assert new_graph.num_nodes() == new_num_nodes
    return new_graph


def create_reduced_recall_val(E, neg_array, num_nodes):
    new_E = []
    new_neg_array = []
    new_nodes = dict.fromkeys(set(range(num_nodes)), True)

    for i in range(len(E)):
        E_src_i, E_dst_i = E[i]
        src_i = new_nodes.get(E_src_i, False)
        dst_i = new_nodes.get(E_dst_i, False)

        if (src_i and dst_i):
            new_E.append([E_src_i, E_dst_i])
            new_neg_array.append(list(neg_array[i]))

    return np.array(new_E), np.array(new_neg_array)

def reduce_neg_array(neg_array, num_nodes, min):
    '''
    I think that each neg_array_list[i] is a list of candidate retrieval nodes.
    A lot of these will be gone with the reduction. I need to reduce each neg_array_list[i] as well

    I guess validation accuracy does not matter too much. Let's see what the smallest vector is.
    '''
    remaining_nodes = dict.fromkeys(set(range(num_nodes)), True)
    new_neg_array = []

    for i in range(len(neg_array)):
        current_neg = [node_i for node_i, exists in enumerate([*map(remaining_nodes.get, neg_array[i])]) if exists][:min]
        new_neg_array.append([*current_neg])
    return np.array(new_neg_array)

def neg_array_min_size(neg_array, num_nodes):
    remaining_nodes = dict.fromkeys(set(range(num_nodes)), True)
    min = np.inf
    max = 0
    for i in range(len(neg_array)):
        curr_len = len([element for element in [*map(remaining_nodes.get, neg_array[i])] if element is not None])
        if curr_len > max:
            max = curr_len
        if curr_len < min:
            min = curr_len

    return min
if __name__ == '__main__':
    # TODO: dgl.DGLGraph.remove_nodes, Yep, I didn't have to make the create_reduced_graph function.
    full_data_root = r'../../data/instance_Pokec_0.1/FullData'
    data_root = r'../../data/instance_Pokec_0.1'
    new_size = .1

    # Reduce train graph
    train_graph = io.load_pickle(osp.join(full_data_root, "train_graph.pkl"))
    reduced_graph = create_reduced_graph(train_graph, k=new_size)
    num_nodes = reduced_graph.num_nodes()
    io.save_pickle(osp.join(data_root, "train_graph.pkl"), reduced_graph)

    # Reduce collate_Graph
    node_collate_graph = io.load_pickle(osp.join(full_data_root, "train_undi_graph.pkl"))
    reduced_graph = create_reduced_graph(node_collate_graph, k=new_size)
    io.save_pickle(osp.join(data_root, "train_undi_graph.pkl"), reduced_graph)


    # Recall: Reduce validation and testing data
    E = io.load_pickle(osp.join(full_data_root, "eval-recall", "recall_val_pos_edgelist.pkl"))
    neg_array = io.load_pickle(osp.join(full_data_root, "eval-recall", "recall_val_neg.pkl"))
    test_neg_array = io.load_pickle(osp.join(full_data_root, "eval-recall", "recall_test_neg.pkl"))
    min_neg_array = min([neg_array_min_size(neg_array, num_nodes), neg_array_min_size(test_neg_array, num_nodes)])
    new_E, new_neg_array = create_reduced_recall_val(E, neg_array, num_nodes)
    new_neg_array = reduce_neg_array(new_neg_array, num_nodes, min_neg_array)
    io.save_pickle(osp.join(data_root, "eval-recall", "recall_val_pos_edgelist.pkl"), new_E)
    io.save_pickle(osp.join(data_root, "eval-recall", "recall_val_neg.pkl"), new_neg_array)
    pos_E = io.load_pickle(osp.join(full_data_root, "eval-recall", "recall_test_pos_edgelist.pkl"))
    new_pos_E, new_test_neg_array = create_reduced_recall_val(pos_E, test_neg_array, num_nodes)
    new_test_neg_array = reduce_neg_array(new_test_neg_array, num_nodes, min_neg_array)
    io.save_pickle(osp.join(data_root, "eval-recall", "recall_test_pos_edgelist.pkl"), new_pos_E)
    io.save_pickle(osp.join(data_root, "eval-recall", "recall_test_neg.pkl"), new_test_neg_array)


    # Rank: Reduce validation and testing data
    # Note: For new_size = .1, rank doesn't find anything in the neg array this remains...
    E = io.load_pickle(osp.join(full_data_root, "eval-rank", "rank_val_pos_edgelist.pkl"))
    neg_array = io.load_pickle(osp.join(full_data_root, "eval-rank", "rank_val_neg.pkl"))
    test_neg_array = io.load_pickle(osp.join(full_data_root, "eval-rank", "rank_test_neg.pkl"))
    new_E, new_neg_array = create_reduced_recall_val(E, neg_array, num_nodes)
    new_pos_E, new_test_neg_array = create_reduced_recall_val(pos_E, test_neg_array, num_nodes)
    min_neg_array = min([neg_array_min_size(neg_array, num_nodes), neg_array_min_size(test_neg_array, num_nodes)])
    new_neg_array = reduce_neg_array(new_neg_array, num_nodes, min_neg_array)
    io.save_pickle(osp.join(data_root, "eval-rank", "rank_val_pos_edgelist.pkl"), new_E)
    io.save_pickle(osp.join(data_root, "eval-rank", "rank_val_neg.pkl"), new_neg_array)
    pos_E = io.load_pickle(osp.join(full_data_root, "eval-rank", "rank_test_pos_edgelist.pkl"))
    new_test_neg_array = reduce_neg_array(new_test_neg_array, num_nodes, min_neg_array)
    io.save_pickle(osp.join(data_root, "eval-rank", "rank_test_pos_edgelist.pkl"), new_pos_E)
    io.save_pickle(osp.join(data_root, "eval-rank", "rank_test_neg.pkl"), new_test_neg_array)


    print('Done.')
