import numpy as np
import networkx as nx


def generate_pnet_files(G1, G2, args, index):
    app = args.app
    dataset = args.dataset
    gt_adj = np.array(nx.adjacency_matrix(G1).todense())

    base_dir = 'results/table2/{}'.format(args.model)+'/{}/{}/'
    # null model (G1)
    np.savetxt((base_dir+'gt-{}.txt').format(dataset, dataset, index),  gt_adj.astype(int), fmt='%s', delimiter=' ')

    # llm consensus network
    adj_G2 = np.array(nx.adjacency_matrix(G2).todense())
    consensus_network_llm = (adj_G2 >= args.consensus).astype(int)
    np.savetxt((base_dir+'recall-{}.txt').format(dataset, dataset, index), ((consensus_network_llm.astype(int)) > 0).astype(int), fmt='%s', delimiter=' ')

