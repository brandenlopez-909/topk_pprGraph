import dgl
from Utils import io
import os.path as osp
import numpy
import torch
import networkx as nx
import time
import pickle

def get_list_of_neighbors(graph,k):

    # The below commented out methods are way to expensive.
    # graph = dgl.to_networkx(graph)
    # neighbors_list = graph.neighbors(k)

    # I'm confused on the difference of predecessors and sucessors.
    # Shouldn't my list be the set of the two? # Either will do the trick
    neighbors_list = set(graph.predecessors(k).tolist()) | set(graph.successors(k).tolist())
    # neighbors_list_2 = set(graph.in_edges(k)[0].tolist()) | set(graph.out_edges(k)[1].tolist())
    return list(neighbors_list)


def connect_nodes(reverse_graph, reverse_node, list_of_nodes):
    """
    Should operate in place.
    """
    connections = list(zip([reverse_node]*len(list_of_nodes), list_of_nodes))
    reverse_graph.add_edges_from(connections)


def build_reverse_graph(graph):
    S_node = [(0, None)]
    # reverse_graph = dgl.DGLGraph() # I don't think DGL supports adding nodes with string names.
    reverse_graph = nx.Graph()
    quant_edges_original_graph = graph.number_of_edges()
    quant_nodes_reversed_graph = 0
    reversed_nodes_list = []
    completed_nodes_list = []

    while len(S_node) != 0 and quant_edges_original_graph != quant_nodes_reversed_graph:
        k,p = S_node.pop()
        neighbors = get_list_of_neighbors(graph,k) # This is where is get's tricky
        for i in neighbors:
            reversed_node_k_i = f'{k},{i}'
            if not reverse_graph.has_node(reversed_node_k_i):
                reverse_graph.add_node(reversed_node_k_i)
            if p!=None:
                reverse_graph.add_edge(reversed_node_k_i, p)
            connect_nodes(reverse_graph, reversed_node_k_i, reversed_nodes_list)
            reversed_nodes_list.append(reversed_node_k_i)

            if i not in completed_nodes_list:
                S_node.append((i, reversed_node_k_i))

        reversed_nodes_list = []
        completed_nodes_list.append(k)
        quant_nodes_reversed_graph = reverse_graph.number_of_nodes()

    return reverse_graph

if __name__ == '__main__':
    train_graph = io.load_pickle(osp.join("../../data/instance_Pokec_0.1", "train_graph.pkl"))


    t0 = time.time()
    reversed_graph = build_reverse_graph(train_graph)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()

    print(f"{t1-t0:.3} Seconds")
    print(f"{(t1-t0)/60:.3} minutes")

    with open('reversed_graph.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(reversed_graph, file)
    file.close()

    with open('reversed_graph_time.txt', 'w') as file:
        file.write(f"{t1-t0:.3} Seconds\n")
        file.write(f"{(t1-t0)/60:.3} minutes")

    file.close()


    print('Done')