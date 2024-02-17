import os
import numpy as np

for dataset in ['fb', 'er', 'author', 'ppi', 'road']:
    # Open a file for each dataset to write the results
    with open(f'{dataset}_num_actors.txt', 'w') as output_file:
        # Set the range to 20 for the 'road' dataset, otherwise 100
        range_limit = 20 if dataset == 'road' else 100

        for index in range(range_limit):
            fname = f'results/table2/gpt-3.5-turbo-1106/{dataset}/{dataset}/gt-{index}.txt'
            arr = np.loadtxt(fname)
            # Write the results to the dataset's file
            output_file.write(f"{dataset} {index} {arr.shape[0]}\n")


