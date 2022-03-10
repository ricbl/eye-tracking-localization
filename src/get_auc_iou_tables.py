from .table_summary_results import get_table
from .get_auc import get_auc
from .iou_results import get_iou
from .get_best_threshold_per_label import get_thresholds
import pandas as pd
from collections import defaultdict
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--test_folder', type=str, nargs='?', default='./runs_paper/test/',
                            help='Folder where test runs are located.')
parser.add_argument('--val_folder', type=str, nargs='?', default='./runs_paper/val/',
                            help='Folder where validation runs are located.')
args = parser.parse_args()
folder_runs_test = args.test_folder +'/'
folder_runs_val = args.val_folder +'/'

def draw_table_latex_per_label(table_filepath, column_name, table_write_path):
    df_iou = pd.read_csv(table_filepath)
    all_iou = {}
    settings = sorted(df_iou['setting'].unique(), reverse = True, key = lambda x: x[-1])
    for setting in settings:
        all_iou[setting] = defaultdict(list)
        df_iou_setting = df_iou[df_iou['setting']==setting]
        for label in sorted(df_iou['label'].unique()):
            df_iou_setting_label = df_iou_setting[df_iou_setting['label']==label]
            # assert(len(df_iou_setting_label)==5)
            for index,row in df_iou_setting_label.iterrows():
                all_iou[setting][label].append(row[column_name])
    txt_file_string = '' 
    txt_file_string += '\\begin{tabular}{|l|c|c|c|} \n \\hline \n Label & Unannotated	&	Ellipses	&	ET model (ours)	\\\\ \\hline \n'
    for label in sorted(df_iou['label'].unique()):
        txt_file_string +=f'{label} & '
        for setting in settings:
            average = np.average(all_iou[setting][label])
            n_models = len(all_iou[setting][label])
            stderror = np.std(all_iou[setting][label])*1.96/math.sqrt(n_models)
            txt_file_string += f'{average:.3f} [{average-stderror:.3f},{average+stderror:.3f}]'
            txt_file_string +=' & '
        txt_file_string +=' \\\\ \n'
    txt_file_string +=' \\hline \n \end{tabular}'
    with open(table_write_path, 'w') as f:
        f.write(txt_file_string)
    return all_iou, settings

def draw_global_auc_iou_table_latex(ious, aucs, settings):
    aggregated = {}
    unaggregated = {}
    unaggregated['IoU'] = ious
    unaggregated['AUC'] = aucs
    txt_file_string = '' 
    txt_file_string += '\\begin{tabular}{|l|c|c|c|} \n \\hline \n Metric & Unannotated	&	Ellipses	&	ET model (ours)	\\\\ \\hline \n'
    for metric in ['AUC', 'IoU']:
        aggregated[metric] = {}
        txt_file_string +=f'{metric} & '
        for setting in settings:
            aggregated[metric][setting] = 0.
            total_labels = 0
            for label in unaggregated[metric][setting]:
                aggregated[metric][setting] += np.array(unaggregated[metric][setting][label])
                total_labels += 1
            aggregated[metric][setting]/=total_labels
            average = np.average(aggregated[metric][setting])
            n_models = len(aggregated[metric][setting])
            stderror = np.std(aggregated[metric][setting])*1.96/math.sqrt(n_models)
            txt_file_string += f'{average:.3f} [{average-stderror:.3f},{average+stderror:.3f}]'
            txt_file_string +=' & '
        txt_file_string +=' \\\\ \n'
    txt_file_string +=' \\hline  \n \end{tabular}'
    with open('Table2.txt', 'w') as f:
        f.write(txt_file_string)
        
get_table(folder_runs_val)
get_thresholds(folder_runs_val)
get_table(folder_runs_test)
get_auc(folder_runs_test)
get_iou(folder_runs_val, folder_runs_test)
ious, settings = draw_table_latex_per_label('./iou_results.csv', 'iou', './Table4.txt')
aucs, settings = draw_table_latex_per_label('./auc.csv', 'auc', './Table3.txt')
draw_global_auc_iou_table_latex(ious, aucs, settings)
