# script used to get a csv table containing which threshold should be used for each class and method (setting)
# when thresholding the output of the model for calculating iou
import numpy as np
import pandas as pd
from .list_labels import list_labels
from .opts import thresholds
import os
from collections import defaultdict

# list_methods represent which method used to calculate model heatmaps
# shoul dbe considered. Options are "ellipse" (using last spatial layer activations)
# or "cam" (using grad-cam)
def get_thresholds(folder_runs, list_methods = ['cam', 'ellipse']):
    folders_runs = []
    for _, dirnames, _ in os.walk(folder_runs):
        folders_runs += dirnames
        break

    runs_folder = folder_runs
    full_df = pd.read_csv('./table_summary_results.csv')
    results_list = []
    for label in list_labels:
        for method in list_methods:
            ious = defaultdict(list)
            folders = []
            settings = []
            original_folders = []
            for index_folder, folder in enumerate(folders_runs):
                folder_df = full_df[full_df['folder']==folder]
                # getting the name of the original training folder to know which setting was used
                with open(f'{runs_folder}/{folder}/log.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line.split(':')[2]=='load_checkpoint_d':
                            original_folder = '/'.join(line.split(':')[3][1:].split('/')[:-1])
                setting = '_'.join(original_folder.split('/')[-1].split('_')[0:-1])
                assert(len(folder_df)==1)
                ious[setting].append([])
                for threshold in thresholds:
                    ious[setting][-1].append(folder_df[f"iou_{label.replace(' ','_')}_val_{method}_iou_{threshold}"])
                folders.append(folder)
                settings.append(setting)
                original_folders.append(original_folder)
            
            # calculate the best iou for each setting
            for setting in ious.keys():
                ious[setting] = np.array(ious[setting])
                ious_averages = ious[setting].mean(axis = 0)[:,0]
                argmax_ious = np.argmax(ious_averages)
                max_ious = ious[setting][:,argmax_ious]
                index_max_iou = 0
                # for all validation folder, check if it belongs to the current setting
                for index_row, _ in enumerate(folders_runs):
                    if settings[index_row] == setting:
                        max_iou = max_ious[index_max_iou]
                        index_max_iou += 1
                        results_list.append({'setting':settings[index_row], 
                            'method':method,
                            'label':label,
                            'max_iou':max_iou[0],
                            'best_threshold':thresholds[argmax_ious],
                            'index_row':index_row,
                            'folder':folders[index_row],
                            'original_folder':original_folders[index_row]})
    pd.DataFrame(results_list).to_csv('best_thresholds.csv', index=False)
