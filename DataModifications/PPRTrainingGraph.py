import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

import os.path as osp
from collections import defaultdict

import dgl
import torch

from Utils import io
from Utils.parse_arguments import parse_arguments



def create_graph(nei, wei, topk=32):
    num_nodes = len(wei)
    # new_graph = dgl.graph(([], []))
    # new_graph.add_nodes(num_nodes)
    U = []
    V = []
    for node_u, v in enumerate(nei):
        U.extend([node_u]*(topk))
        V.extend(v.tolist())
    U_undi = U.copy()
    V_undi = V.copy()
    graph = dgl.graph((U, V))

    U_undi.extend(V), V_undi.extend(U)
    undirected_graph = dgl.graph((U_undi, V_undi))
    return graph, undirected_graph


def main():

    parsed_results = parse_arguments()
    config = defaultdict(int)
    config.update(parsed_results)

    OUTPUT_ROOT = config['results_root']
    TOPK = config['topk']
    PPR_DATA_ROOT = osp.join(OUTPUT_ROOT, 'ppr/undirected-top100')
    DATA_ROOT = osp.join(OUTPUT_ROOT, config['model'])
    # Pokec has an average of 12 edges per node, LiveJournal is 16

    raw_nei = io.load_pickle(osp.join(PPR_DATA_ROOT, "nei.pkl"))
    raw_wei = io.load_pickle(osp.join(PPR_DATA_ROOT, "wei.pkl"))

    nei = torch.LongTensor(raw_nei[:, 1: TOPK + 1])
    wei = torch.FloatTensor(raw_wei[:, 1: TOPK + 1])

    train_graph, train_undi_graph = create_graph(nei, wei, TOPK)

    io.save_pickle(osp.join(DATA_ROOT, 'train_graph.pkl'), train_graph)
    io.save_pickle(osp.join(DATA_ROOT, 'train_undi_graph.pkl'), train_undi_graph)

if __name__ == "__main__":
    main()
    print("Completed PPR Training Graph")