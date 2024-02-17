import argparse
import sys
import datetime
import networkx as nx
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from itertools import combinations
import logging
import time
import os


def set_up():
    args, sys_argv = get_args()
    logger = set_up_logger(args, sys_argv)
    # set_random_seed(args.seed)
    return args, logger


def get_args():

    parser = argparse.ArgumentParser('Interface for testing microstructures and accuracy of graph recall by LLMs')

    # key parameters: dataset, features, model, and sampling budget
    parser.add_argument('--seed', type=int, default=2, help='ramdom seed for networkx drawing, recommend 2 for irreducible, 3 for reducible')
    parser.add_argument('--cap', type=int, default=100, help='a sub-sample of the batched vignettes')

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='gpt-3.5-turbo, gpt-4, gemini-pro')
    parser.add_argument('--app', type=str, default='fb', help='application domains to study', choices=['fb', 'road', 'ppi', 'author', 'er'])
    parser.add_argument('--dataset', type=str, default='fb', help='dataset to us', choices=['fb', 'road', 'ppi', 'author', 'er', 'iw','is', 'rw','rs'])
    # road: cap=20
    parser.add_argument('--memclear', type=int, default=7, help='measure clear strength, measured in number of sets of 3-sentences')
    parser.add_argument('--pnet', type=int, default=1, help='whether to generate intermediate pnet files for ERGM estimation')


    # brashear's papers
    parser.add_argument('--type', type=str, default='iw', help='irreducible weak, irreducible strong, reducible weak, reducible strong')
    parser.add_argument('--consensus', type=float, default=0.1, help='consensus level for filtering the network')
    parser.add_argument('--sex', type=int, default=0, help='0: none, 1: male, 2: female')

    parser.add_argument('--permuted', type=int, default=0, help='whether to permute the edges in prompt (query)')
    parser.add_argument('--max_tokens', type=int, default=4096, help='max_tokens per request')


    # other system parameters
    parser.add_argument('--log_dir', type=str, default='./log/', help='gpt-3.5-turbo, gpt-4')

    # for gemini repetition
    parser.add_argument('--repetition', type=int, default=1, help='how many times to repeat the same prompt for gemini')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    # fix gpt models
    if args.model == 'gpt-3.5-turbo':
        args.model = 'gpt-3.5-turbo-1106' # Using 1106 version by default
    elif args.model == 'gpt-4':
        args.model = 'gpt-4-1106-preview'  # Using 1106 version by default
    else:
        pass
    return args, sys.argv


def set_up_logger(args, sys_argv):
    # set up running log
    runtime_id = '{}-{}-{}-{}-{}'.format(str(args.app), args.model, args.sex, args.memclear,  str(time.time()))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_dir = '{}/{}/'.format(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = log_dir + runtime_id + '.log'
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger


def now():
    now = datetime.datetime.now()
    return int(now.timestamp())


def print_messages(messages, logger):
    logger.info('='*50+'messages'+'='*50)
    for message in messages:
        logger.info(message['role'])

        logger.info(message['content'])
        logger.info('\n')
    logger.info('='*100)


def print_response(completion, logger):
    response = completion.choices[0].message.content.strip()
    logger.info(response)


def load_string(filename):
    with open(filename, encoding='utf-8') as f:
        string = f.read()
    return string


def concatenate_segments(segments):
    coin = 0
    prompt = ''
    roles = ['User: ', 'Assistant: ']
    for seg in segments:
        prompt = prompt + roles[coin] + seg
        coin = 1-coin
    return prompt


def compute_metrics(G1: nx.Graph, G2: nx.Graph):
    gt = []
    pred = []
    edge2weight = nx.get_edge_attributes(G2, 'weight')
    for edge in combinations(G1.nodes, 2):
        gt.append(G1.has_edge(*edge))
        if edge in edge2weight:
            pred.append(int(edge2weight[edge]>0.5))
        else:
            pred.append(0)

    gt, pred = np.array(gt), np.array(pred)

    return f1_score(gt, pred), accuracy_score(gt, pred), precision_score(gt, pred), recall_score(gt, pred)


def args2str(args):
    args_str = ' '.join(f'{key}={value}' for key, value in vars(args).items())
    return args_str


def edge2str(edges):
    return ', '.join([f"({u}, {v})" for u, v in edges])