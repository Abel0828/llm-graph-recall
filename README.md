# Microstructures and Accuracy of Graph Recall by LLMs

## Environment & Setup
* `python>=3.9`
* `openai==1.8.0`
* `google-generativeai==0.3.2`
* `tiktoken==1.5.2`
* `numpy`, `scikit-learn`, `networkx`, `tqdm`
* Replace [here](./network_recall.py#L8) with your own `openai_api_key`
* Replace [here](./network_recall_gemini.py#L14) with your own `google_api_key`

## Command Examples
- Running on a single graph sample of the Facebook dataset, using GPT-3.5:\
`python network_recall.py --dataset fb --cap 1`
- Running on the full Facebook dataset with co-authorship as narrative style, 5 as the memory clearance strength, using GPT-4:\
`python network_recall.py --dataset fb --app author --model gpt-4 --memclear 5`


## Usage Summary
```
Interface for testing microstructures and accuracy of graph recall by LLMs [-h] [--seed SEED] [--cap CAP] [--model MODEL] [--app {fb,road,ppi,author,er}] [--dataset {fb,road,ppi,author,er,iw,is,rw,rs}] [--memclear MEMCLEAR]
                                                                                  [--pnet PNET] [--type TYPE] [--consensus CONSENSUS] [--sex SEX] [--permuted PERMUTED] [--max_tokens MAX_TOKENS] [--log_dir LOG_DIR] [--repetition REPETITION]
```

## Optinal Arguments
```
  -h, --help            show this help message and exit
  --seed SEED           ramdom seed for networkx drawing, recommend 2 for irreducible, 3 for reducible
  --cap CAP             a sub-sample of the batched vignettes
  --model MODEL         gpt-3.5-turbo, gpt-4, gemini-pro
  --app {fb,road,ppi,author,er}
                        application domains to study
  --dataset {fb,road,ppi,author,er,iw,is,rw,rs}
                        dataset to us
  --memclear MEMCLEAR   measure clear strength, measured in number of sets of 3-sentences
  --pnet PNET           whether to generate intermediate pnet files for ERGM estimation
  --type TYPE           irreducible weak, irreducible strong, reducible weak, reducible strong
  --consensus CONSENSUS
                        consensus level for filtering the network
  --sex SEX             0: none, 1: male, 2: female
  --permuted PERMUTED   whether to permute the edges in prompt (query)
  --max_tokens MAX_TOKENS
                        max_tokens per request
  --log_dir LOG_DIR     gpt-3.5-turbo, gpt-4
  --repetition REPETITION
                        how many times to repeat the same prompt for gemin
```

## Authors
[Yanbang Wang](https://www.cs.cornell.edu/~ywangdr/), [Hejie Cui](https://hejiecui.com/), [Jon Kleinberg](https://www.cs.cornell.edu/home/kleinber/) (Cornell University, Emory University)
