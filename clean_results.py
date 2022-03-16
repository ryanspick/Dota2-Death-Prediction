

import sys
import os
import glob
import commentjson

RESULTS_PATH = "/users/rjs623/scratch/esport/cluter_results"
RESULT_NAME = "results_10_new.pickle"

import subprocess
def tail(file, n):
    with open(file) as f:
        content = f.readlines()
    return content[-n:]

log_files = glob.glob(RESULTS_PATH + "/**/*.log", recursive=True)
print(len(log_files))
dirs_with_results = [ os.path.dirname(os.path.abspath(log)) for log in log_files]

print(log_files[108])

# look in the log file, for the last "Epoch done"
# look at the loss and the accuracy

data_points = []

for i,res_dir in enumerate(dirs_with_results):
    
    print(log_files[i])
    last_log_lines = tail(log_files[i],150)
    accuracy_found = False

    accuracy = None
    loss = None
    num_epoch = None

    for line in last_log_lines:
        #example: Epoch done  7623  loss:  2.397298  accuracy:  0.09747869318181818
        if "Epoch done" in line:
            split_line = line.split()
            accuracy = float(split_line[6])
            loss = float(split_line[4])
            num_epoch = int(split_line[2])
            accuracy_found = True
            
            break

    if accuracy_found == True:
        with open(res_dir + "/config.json") as f:
            config = commentjson.load(f)
            data_points.append((accuracy,loss,num_epoch,config,log_files[i]))
    else:
        print("No accuracy found: ",res_dir)

import pickle

with open(RESULT_NAME, 'wb') as f:
    pickle.dump(data_points, f)


    




