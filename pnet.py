import numpy as np
import networkx as nx
import os


def generate_pnet_files(G1, G2, args, index):
    gt_adj = np.array(nx.adjacency_matrix(G1).todense())

    base_dir = 'results/table2/' if args.dataset not in ['i', 'r'] else 'results/table1-{}/'.format(args.consensus)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    model_dir = base_dir+args.model+'/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    dataset_dir = model_dir + args.dataset +'/'
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    app_dir = dataset_dir + args.app + '/'
    if not os.path.exists(app_dir):
        os.mkdir(app_dir)

    # null model (G1)

    np.savetxt((app_dir+'gt-{}.txt').format(index),  gt_adj.astype(int), fmt='%s', delimiter=' ')

    # llm consensus network
    adj_G2 = np.array(nx.adjacency_matrix(G2).todense())
    consensus_network_llm = (adj_G2 >= args.consensus).astype(int)
    np.savetxt((app_dir+'recall-{}.txt').format(index), ((consensus_network_llm.astype(int)) > 0).astype(int), fmt='%s', delimiter=' ')
    print('Pnet file {} saved at '.format(index), app_dir)

