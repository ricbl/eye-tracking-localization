# script used to get the iou values for each test run, using the best thresholds calculated 
# from the validation runs.
# The script get_best_threshold_per_label has to be run with the associated validation runs 
# to generate the best_thresholds.csv needed for this script.
# table_summary_results.csv should have been generated with table_summary_results.py (using the test runs).
import pandas as pd
import os

def get_iou(val_run_folder, test_runs_folder, method = 'cam'):
    df_thresholds = pd.read_csv(f'./best_thresholds_{val_run_folder.replace("/", "|")}.csv')
    table_summary = pd.read_csv(f'./table_summary_results_{test_runs_folder.replace("/", "|")}.csv')
    df_thresholds = df_thresholds[df_thresholds['method']==method]
    test_folders = {}
    folders_runs = []
    for _, dirnames, _ in os.walk(test_runs_folder):
        folders_runs += dirnames
        break
    
    # associate each test run folder to the original training run folder
    for index_folder, folder in enumerate(folders_runs):
        with open(f'{test_runs_folder}/{folder}/log.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.split(':')[2]=='load_checkpoint_d':
                    original_folder = '/'.join(line.split(':')[3][1:].split('/')[:-1])
                    break
        test_folders[original_folder] = folder
    
    # get the best threshold of each original_folder, getting the test score for that threshold
    iou_results = []
    for index, row in df_thresholds.iterrows():
        original_folder = row['original_folder']
        label = row['label']
        setting = row['setting']
        threshold = row['best_threshold']
        row_test = table_summary[table_summary['folder']==test_folders[original_folder]]
        assert(len(row_test)==1)
        if threshold == 0:
            threshold = int(threshold)
        iou = row_test[f'iou_{label.replace(" ","_")}_val_{method}_iou_{threshold}'].values[0]
        iou_results.append({'original_folder':original_folder,'folder':folder,'setting':setting,'label':label,'threshold':threshold,'iou':iou})
    
    pd.DataFrame(iou_results).to_csv(f'iou_results_{method}_{test_runs_folder.replace("/", "|")}.csv', index=False)