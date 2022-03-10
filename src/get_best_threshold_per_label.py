import numpy as np
import pandas as pd
from .list_labels import list_labels
from .opts import thresholds
import os

def get_thresholds(folder_runs):
    folders_runs = []
    for _, dirnames, _ in os.walk(folder_runs):
        folders_runs += dirnames
        break

    runs_folder = folder_runs
    full_df = pd.read_csv('./table_summary_results.csv')
    results_list = []
    for label in list_labels:
        for method in ['cam', 'ellipse']:
            ious = []
            folders = []
            settings = []
            for index_folder, folder in enumerate(folders_runs):
                folder_df = full_df[full_df['folder']==folder]
                with open(f'{runs_folder}/{folder}/log.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line.split(':')[2]=='load_checkpoint_d':
                            original_folder = '/'.join(line.split(':')[3][1:].split('/')[:-1])
                with open(f'{original_folder}/log.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line.split(':')[2]=='percentage_annotated':
                            percentage_annotated = line.split(':')[3][1:]
                        if line.split(':')[2]=='use_et':
                            use_et = line.split(':')[3][1:]
                        if line.split(':')[2]=='dataset_type':
                            dataset_type = line.split(':')[3][1:]
                
                dataset_used = 'ellipse' if use_et=='False' else dataset_type
                setting = f'{percentage_annotated}_{dataset_used}'
                assert(len(folder_df)==1)
                ious.append([])
                for threshold in thresholds:
                    ious[index_folder].append(folder_df[f"iou_{label.replace(' ','_')}_val_{method}_iou_{threshold}"])
                folders.append(folder)
                settings.append(setting)
            ious = np.array(ious)
            ious_averages = ious.mean(axis = 0)[:,0]
            argmax_ious = np.argmax(ious_averages)
            max_ious = ious[:,argmax_ious]
            for index_row, max_iou in enumerate(max_ious):
                results_list.append({'setting':settings[index_row], 'method':method,'label':label, 'max_iou':max_iou[0], 'best_threshold':thresholds[argmax_ious],'index_row':index_row, 'folder':folders[index_row]})
    pd.DataFrame(results_list).to_csv('best_thresholds.csv', index=False)
