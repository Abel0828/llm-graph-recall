# To do:
# check prompt and messages, save pnet files, logging,
# prompting, is intro appropriate?
# run batch script, issue with naming dirctory by model name

import os
import pickle
    
import json
import google.generativeai as genai

# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
# GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = '' # to replace with your own google_api_key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
# chat = model.start_chat()

import re
import matplotlib.pyplot as plt
from utils import *
from pnet_gemini import *
from constants import *
from api import *

import random
import time
# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_your_.json_credential_file"

def generate_all_pairs(names, permuted=False, topk=-1, joint=[',', '),('], prefix='(', suffix=')'):
    strs = []
    for pair in combinations(names, 2):
        s = joint[0].join(pair)
        strs.append(s)
    if permuted:
        strs = np.random.permutation(strs)
    if topk > 0:
        strs = strs[:topk]
    return prefix + joint[1].join(strs) + suffix


def get_response(completion, index_recall_start, index_recall_end):
    tokens = completion['choices'][0]['logprobs']['tokens'][index_recall_start: index_recall_end]
    token_logprobs = completion['choices'][0]['logprobs']['token_logprobs'][index_recall_start: index_recall_end]
    return tokens, token_logprobs



def get_fname(t):
    type2fname = {'iw': 'irreducible weak', 'is': 'irreducible strong', 'rw': 'reducible weak',
                  'rs': 'reducible strong'}
    fname = type2fname[t]
    return fname


def make_graph(nodes, edges=None, edge2prob=None):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    if edge2prob is None:
        G.add_edges_from(edges)
    else:
        G.add_weighted_edges_from([(*key, value) for key, value in edge2prob.items()])
    return G


# for batched evaluations
def create_prompt_batch(args, dic):
    names = dic['nodes']
    # logger.info('Number of nodes: {}'.format(len(names)))
    # if not isinstance(names, str):
    #     names = ', '.join(map(str, names))
    # logger.info('Node names:'+', '.join(names))
    generalintro = load_string(prompts_dir + 'intro-{}.txt'.format({0: 'general',1: 'male', 2: 'female'}[args.sex]))
    generalintroresponse = load_string(prompts_dir + 'introresponse-general.txt')
    vignette = dic['vignette'] +'\n\n'
    question = load_string(prompts_dir + 'question-general.txt') + generate_all_pairs(names, permuted=args.permuted) +'\n\n'

    segments = [generalintro, generalintroresponse]
    if args.memclear > 0:
        mcintro = load_string(prompts_dir+'memoryclear/intro.txt')
        mcintroresponse = load_string(prompts_dir + 'memoryclear/introresponse.txt')
        mc = load_string(prompts_dir + 'memoryclear/memory-clear-{}.txt'.format(args.memclear))
        segments += [vignette+mcintro, mcintroresponse+mc, question]
    else:
        segments += [vignette+question]

    # segments += ['']
    prompt = concatenate_segments(segments)
    # print(prompt)
    return prompt

def create_prompt(args):
    generalintro = load_string(prompts_dir + 'intro.txt')
    generalintroresponse = load_string(prompts_dir + 'introresponse.txt')
    segments = [generalintro, generalintroresponse]
    vignette = load_string(prompts_dir + 'vignettes/{}.txt'.format(expand[args.type]))
    question = load_string(prompts_dir + 'question.txt') + generate_all_pairs(names, permuted=args.permuted) +'\n\n'

    if args.sex > 0:
        sex = {1: 'male', 2: 'female'}[args.sex]
        sexquestion = load_string(prompts_dir+'sex/{}question.txt'.format(sex))
        sexresponse = load_string(prompts_dir + 'sex/{}response.txt'.format(sex))

        segments += [sexquestion, sexresponse]
    if args.memclear > 0:
        mcintro = load_string(prompts_dir+'memoryclear/intro.txt')
        mcintroresponse = load_string(prompts_dir + 'memoryclear/introresponse.txt')
        mc = load_string(prompts_dir + 'memoryclear/memory-clear-{}.txt'.format(args.memclear))
        segments += [vignette+mcintro, mcintroresponse+mc, question]
    else:
        segments += [vignette+question]

    prompt = concatenate_segments(segments)
    return prompt

def hash_prompt(model, prompt):
    return model + str(len(prompt)) + prompt[len(prompt)//2-1:len(prompt)//2+2]

# def create_chatcompletion(messages, args):
#     completion = client.chat.completions.create(
#         model=args.model,
#         logprobs=True,
#         top_logprobs=5,
#         temperature=0,
#         seed=0,
#         messages=messages
#     )
#     return completion

def create_response(messages, args):
    # return model.generate_content(messages, args)
    # Restructure the `messages` list to meet the API's expectations:
    structured_messages = [
        # {'role': message['role'], 'parts': [{'text': message['content']}]} for message in messages[:-1]
        {'role': message['role'], 'parts': {'text': message['content']}} for message in messages[1:-1]
    ]

    json_string = json.dumps(structured_messages, indent=4)  # Add indentation for readability
    # print("json:\n", json_string)

    user_input = messages[-1]['content']
    # print("user input:\n", user_input)
    # print(user_input)
    chat = model.start_chat(history=structured_messages)

    try:
        response = chat.send_message(user_input)
        return response
    except:
        print("error")
        return ""
    

# def prompt2messages(prompt):
#     messages = [{"role": "system", "content": "You are a large language model trained by OpenAI. Answer as concisely as possible."}]
#     exchanges = re.split(r"(User:|Assistant:)", prompt)
#     # The first element of the split list will be an empty string, so remove it
#     if exchanges[0].strip() == '':
#         exchanges = exchanges[1:]
#     # Pair up the exchanges and strip whitespace
#     for i in range(0, len(exchanges), 2):
#         speaker = exchanges[i].strip(":").lower()
#         speech = exchanges[i + 1].strip()
#         messages.append({"role": speaker, "content": speech})
#     return messages

def prompt2messages(prompt):
    messages = [{"role": "system", "content": "You are a large language model trained by OpenAI. Answer as concisely as possible."}]
    exchanges = re.split(r"(User:|Assistant:)", prompt)
    # The first element of the split list will be an empty string, so remove it
    if exchanges[0].strip() == '':
        exchanges = exchanges[1:]
    # Pair up the exchanges and strip whitespace
    for i in range(0, len(exchanges), 2):
        speaker = exchanges[i].strip(":").lower()
        if speaker == 'assistant':
            speaker = 'model'
        speech = exchanges[i + 1].strip()
        messages.append({"role": speaker, "content": speech})
    return messages

# def get_completion(prompt, args):
#     cahce_name = 'cache/all.pkl'
#     key = hash_prompt(args.model, prompt)
#     messages = prompt2messages(prompt)
#     print_messages(messages, logger)
#     logger.info('number of tokens in request: ' + str(num_tokens_from_messages(messages, args.model)))
#     update = 1
#     try:
#         with open(cahce_name, 'rb') as f:
#             cache = pickle.load(f)
#             if key in cache:
#                 completion = cache[key][0]
#                 assert(prompt == cache[key][1])
#                 update = 0
#             else:
#                 completion = create_chatcompletion(messages, args)
#                 cache.update({key: (completion, prompt)})
#             logger.info('cache found!')
#     except:
#         logger.info('cache not found, sending request ...')
#         completion = create_chatcompletion(messages, args)
#         cache = {key: (completion, prompt)}
#     if update:
#         with open(cahce_name, 'wb') as f:
#             pickle.dump(cache, f)
#     return completion

def get_response(prompt, args):
    cahce_name = 'cache/all_gemini.pkl'
    key = hash_prompt(args.model, prompt)
    messages = prompt2messages(prompt)
    # print_messages(messages, logger)
    # logger.info('number of tokens in request: ' + str(num_tokens_from_messages(messages, args.model)))
    update = 1
    try:
        with open(cahce_name, 'rb') as f:
            cache = pickle.load(f)
            if key in cache:
                response = cache[key][0]
                assert(prompt == cache[key][1])
                update = 0
            else:
                response = create_response(messages, args)
                cache.update({key: (response, prompt)})
            logger.info('cache found!')
    except:
        logger.info('cache not found, sending request ...')
        response = create_response(messages, args)
        cache = {key: (response, prompt)}
    if update:
        with open(cahce_name, 'wb') as f:
            pickle.dump(cache, f)
    return response


def check_format(s):
    # pattern = r'\([A-Za-z0-9 ]+, ?[A-Za-z0-9 ]+\): [01]'
    # pattern = r'\([A-Za-z0-9 !@#$%^&*()_+=\-[\]{};:\'",<.>/?`~|\\]+, ?[A-Za-z0-9 !@#$%^&*()_+=\-[\]{};:\'",<.>/?`~|\\]+\): [01]'
    # pattern = r'\([\w !@#$%^&*()_+=\-[\]{};:\'",<.>/?`~|\\]+, ?[\w !@#$%^&*()_+=\-[\]{};:\'",<.>/?`~|\\]+\): [01]'
    pattern = r'\(.+,.+\): [01]'
    return bool(re.fullmatch(pattern, s))

# def get_edge2prob(completion, nodes):
#     logprobs_json = completion.choices[0].logprobs
#     response = completion.choices[0].message.content
#     p_l = []
#     for token_dic in logprobs_json.content:
#         token, token_logprob = token_dic.token.strip(), token_dic.logprob
#         if token in ['1', '0']:
#             top_logprobs = token_dic.top_logprobs
#             alts = {}
#             for TopLogprob in top_logprobs:
#                 alt_token, alt_logprob = TopLogprob.token, TopLogprob.logprob
#                 alts[alt_token] = alt_logprob
#             opposite = '0' if token == '1' else '1'
#             if opposite in alts: # need to normalize
#                 opposite_logprob = alts[opposite]
#                 p_l.append((int(token.strip()), np.exp(token_logprob)/(np.exp(token_logprob)+ np.exp(opposite_logprob))))
#             else:
#                 p_l.append((int(token.strip()), np.exp(token_logprob)))

#     edge2prob = {}
#     for edge, p_item in zip(combinations(nodes, 2), p_l):
#         if p_item[0] == 1:
#             edge2prob[edge] = p_item[1]
#         else:
#             edge2prob[edge] = 1.0-p_item[1]
#     return edge2prob

def get_edge2prob(filename):
    pair_counts = {}
    base_dir = f'results/table2/{args.model}/{args.dataset}/response_text/'
    with open(base_dir+filename, 'r') as f:
        for line in f:
            # print(line.strip())
            if not check_format(line.strip()):
                print("format error")
                print(line.strip())
                continue
            # pattern = r'\([A-Za-z]+, [A-Za-z]+\): [01]'
            # pairs = re.findall(pattern, line.strip())
            try:
                name_pair, value = line.strip().split(": ")
            except:
                print(line.strip())
                continue
            try:
                name_pair = tuple(name_pair.strip("()").split(","))
                value = int(value)
            # name_pair = tuple(name_pair.strip("()").split(","))
            # value = int(value)
            except:
                print(line.strip())
                continue
            # for pair in pairs:
            #     name_pair, value = pair.split(": ")
            #     value = int(value)

            if name_pair not in pair_counts:
                pair_counts[name_pair] = {'total': 0, 'ones': 0}
            pair_counts[name_pair]['total'] += 1
            pair_counts[name_pair]['ones'] += value

    edge2prob = {pair: counts['ones']/counts['total'] for pair, counts in pair_counts.items()}

    return edge2prob


def discretize_graph(G2):
    G = nx.Graph()
    edge2weight = nx.get_edge_attributes(G2, 'weight')
    for edge in combinations(G2.nodes, 2):
        if edge in edge2weight and edge2weight[edge]>0.5:
            G.add_edge(*edge)
    return G


def brashear_eval(args):
    prompt = create_prompt(args)
    # completion = get_completion(prompt, args)
    response = get_response(prompt, args)
    # logger.info(completion.choices[0].message.content)  # print response
    # logger.info(response.text)  # print response
    edge2prob = get_edge2prob(response, names)
    return edge2prob


def batch_eval(args):
    with open('data/preprocessed/{}_vignette.pkl'.format(args.dataset), 'rb') as f:
        node_vignette = pickle.load(f)

    f1_scores = []
    repetition_num = args.repetition
    for i, (node_name, dic) in enumerate(list(node_vignette.items())[:args.cap]):
        filename = f"{node_name}.text"
        print(i)
        print(filename)
        base_dir = f'results/table2/{args.model}/{args.dataset}/response_text/'
        with open(base_dir+filename, 'w') as f:
            for _ in range(repetition_num):
                prompt = create_prompt_batch(args, dic)
                response = get_response(prompt, args)
                if response == "":
                    print("no response")
                    continue
                response_text = response.text
                f.write(response_text+ "\n")
                # Random sleep for 1 to 2 seconds
                sleep_time = random.uniform(1, 2)
                # time.sleep(sleep_time)

        edge2prob = get_edge2prob(filename)

        G1 = make_graph(dic['nodes'], edges=dic['graph'])
        G2 = make_graph(dic['nodes'], edge2prob=edge2prob)

        if args.pnet:
            generate_pnet_files(G1, G2, args, index=i)
        f1, acc, precision, recall = compute_metrics(G1, G2)

        result = {(node_name, args2str(args)): {'G1': G1, 'G2': G2, 'response': response, 'f1': f1, 'acc': acc, 'precision': precision, 'recall': recall}}
        # save_one_result(result)

        f1_scores.append(f1)
        # logger.info(node_name)
        logger.info('ground truth: ' + edge2str(sorted(G1.edges)))
        logger.info('recalled: ' + edge2str(sorted(discretize_graph(G2).edges)))
        logger.info('F1 score: {:.4f}'.format(f1))
        logger.info('*'*150)

    return f1_scores


def save_one_result(result):
    cahce_name = 'cache/results_cache.pkl'
    if not os.path.exists(cahce_name):
        with open(cahce_name, 'wb') as f:
            pickle.dump({}, f)
    with open(cahce_name, 'rb') as f:
        cache = pickle.load(f)
    cache.update(result)
    with open(cahce_name, 'wb') as f:
        pickle.dump(cache, f)


def batch(args):
    if args.dataset in ['iw', 'is', 'rw', 'rs']:
        brashear_eval(args)
    else:
        batch_eval(args)


def read_humans(args):
    name = []
    if 'i' in args.type:
        name.append('Irreducible')
    if 'r' in args.type:
        name.append('Reducible')

    if 's' in args.type:
        name.append('Strong')
    if 'w' in args.type:
        name.append('Weak')
    human_names = list(tuple(names))
    human_names[3], human_names[4] = human_names[4], human_names[3]

    dir = 'results/Consensus Networks/'+ ' '.join(name)+'/'
    fnames = [x for x in os.listdir(dir) if 'consensus' in x]
    level_network = []
    for fname in fnames:
        with open(dir+fname, 'r') as f:
            fname = fname[:-4]
            if '.' in fname:
                level = float('0.'+fname.strip().split('.')[-1])
            else:
                level = 1.0
            lines = []
            for line in f.readlines():
                lines.append([int(x) for x in line.strip().split('\t')])
            adj = np.array(lines)
            level_network.append((level, adj))
    level_network = sorted(level_network)

    # read edge2prob from level_network
    prev = None
    prev_level = 0.0
    edge2prob = {}
    for level, adj in level_network:
        if prev is None:
            prev = adj
            prev_level = level
        else:
            diff = prev - adj
            assert((diff>=0).all())
            for i in range(diff.shape[0]):
                for j in range(i+1, diff.shape[1]):
                    if diff[i,j] > 0:
                        edge2prob[human_names[i], human_names[j]] = prev_level
            prev = adj
            prev_level = level

    for i in range(len(human_names)):
        for j in range(i + 1, len(human_names)):
            edge = (human_names[i], human_names[j])
            if edge not in edge2prob:
                edge2prob[edge] = 0.0
    return edge2prob


def brashear(args):
    if 'r' in args.type:
        args.seed = 3
    fname = get_fname(args.type)
    # ground truths
    G1 = nx.Graph()
    G1.add_nodes_from(nodes)
    if 'irreducible' in fname:
        G1.add_edges_from(gt_i)
    else:
        G1.add_edges_from(gt_r)
    seed = args.seed
    pos = nx.spring_layout(G1, seed=seed)
    edge_width = 2.0
    plt.figure(figsize=(10,12), dpi=200)
    plt.subplot(221)
    nx.draw_networkx(G1, pos, with_labels=True, node_color='lightblue', connectionstyle='arc3,rad=0.1', font_size=8, width=edge_width)
    plt.title("Ground truth - {}".format(args.type))

    # gpt
    edge2prob = brashear_eval(args)
    G2 = nx.Graph()
    G2.add_nodes_from(nodes)
    G2.add_weighted_edges_from([(*key, value) for key, value in edge2prob.items()])

    plt.subplot(222)
    edge_alphas = [(weight / 1.0) for weight in nx.get_edge_attributes(G2, 'weight').values()]
    nx.draw_networkx(G2, pos, with_labels=True, node_color='lightgreen', edge_color=[(0, 0, 0, alpha) for alpha in edge_alphas],
                     connectionstyle='arc3,rad=0.1', font_size=8, width=edge_width)
    f1 = compute_metrics(G1, G2)[0]
    title = "{} answers {} (F1:{:.3f},MemClr={},Sex={})".format(args.model, args.type, f1, args.memclear, args.sex)
    logger.info(title)
    plt.title(title)



    # human
    edge2prob4 = read_humans(args)
    G4 = nx.Graph()
    G4.add_nodes_from(nodes)
    G4.add_weighted_edges_from([(*key, value) for key, value in edge2prob4.items()])

    plt.subplot(224)
    edge_alphas4 = [(weight / 1.0) for weight in nx.get_edge_attributes(G4, 'weight').values()]
    nx.draw_networkx(G4, pos, with_labels=True, node_color='lightgreen', edge_color=[(0, 0, 0, alpha) for alpha in edge_alphas4],
                     connectionstyle='arc3,rad=0.1', font_size=8, width=edge_width)
    f1 = compute_metrics(G1, G4)[0]
    plt.title("Human answers {} (F1:{:.3f},MemClr={},Sex={})".format(args.type, f1, args.memclear, args.sex))

    # for motif analysis in PNet
    generate_network_and_structural_zero_files(G1, G2, G4, args, consensus=args.consensus)


    # human
    # edge2prob4 = read_humans(args)
    G44 = nx.Graph()
    G44.add_nodes_from(nodes)
    G44.add_weighted_edges_from([(*key, value) for key, value in edge2prob4.items() if value>=0.5])

    plt.subplot(223)
    edge_alphas44 = [(weight / 1.0) for weight in nx.get_edge_attributes(G44, 'weight').values()]
    nx.draw_networkx(G44, pos, with_labels=True, node_color='lightgreen', edge_color=[(0, 0, 0, alpha) for alpha in edge_alphas44],
                     connectionstyle='arc3,rad=0.1', font_size=8, width=edge_width)
    f1 = compute_metrics(G1, G44)[0]
    plt.title("Human answers {} (F1:{:.3f},MemClr={},Sex={})".format(args.type, f1, args.memclear, args.sex))


    plt.tight_layout()
    plt.savefig('figs/{}_{}.png'.format(fname, now()))



    # plt.show()

def generate_network_and_structural_zero_files(G1, G2, G3, args, consensus=0.10):
    # structural zeros
    gt_adj = np.array(nx.adjacency_matrix(G1).todense())
    structural_zeros = 1-gt_adj
    structural_zeros = structural_zeros.astype(int)
    # np.savetxt('pnet_files/structuralzeros-{}.txt'.format(args.type), structural_zeros, fmt='%s', delimiter=' ')


    # null model
    # np.savetxt('pnet_files/null-{}.txt'.format(args.type), gt_adj.astype(int), fmt='%s', delimiter=' ')


    # llm consensus network
    adj_G2 = np.array(nx.adjacency_matrix(G2).todense())
    consensus_network_llm = (adj_G2 >= consensus).astype(int)
    np.savetxt('pnet_files/network-llm-{}.txt'.format(args.type), ((consensus_network_llm+gt_adj.astype(int))>0).astype(int), fmt='%s', delimiter=' ')

    # human consensus network
    adj_G3 = np.array(nx.adjacency_matrix(G3).todense())
    consensus_network_human = (adj_G3 >= consensus).astype(int)
    np.savetxt('pnet_files/network-human-{}.txt'.format(args.type), ((consensus_network_human+gt_adj.astype(int))>0).astype(int), fmt='%s', delimiter=' ')



args, logger = set_up()
batch(args)
# brashear(args)


