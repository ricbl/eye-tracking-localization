# script used to calculate the tables included in the paper
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
folder_runs_test = args.test_folder + '/'
folder_runs_val = args.val_folder + '/'

# function used to format the either the calculated aucs or ious, with one row per label, to the format of a latex table
def draw_table_latex_per_label(table_filepath_u, table_filepath_a, column_name, table_write_path):
    df_iou_u = pd.read_csv(table_filepath_u)
    df_iou_a = pd.read_csv(table_filepath_a)
    all_iou = {}
    #the setting reported in the paper were: unannotated, ellipse and ET data
    settings = ['unannotated_baseline', 'ellipses_baseline', 'et_data_model']
    settings = {'test_unet_300_li_loss_3_unannotated':df_iou_u, 'test_unet_300_li_loss_3_ellipse':df_iou_a, 'test_unet_300_li_loss_3_et':df_iou_a}    
    for setting in settings:
        all_iou[setting] = defaultdict(list)
        df_iou_setting = settings[setting][settings[setting]['setting']==setting]
        for label in sorted(settings[setting]['label'].unique()):
            df_iou_setting_label = df_iou_setting[df_iou_setting['label']==label]
            for index,row in df_iou_setting_label.iterrows():
                # separating the iou of each setting and label into separate lists
                all_iou[setting][label].append(row[column_name])
    txt_file_string = '' 
    txt_file_string += '\\begin{tabular}{|l|c|c|c|} \n \\hline \n Label & Unannotated	&	Ellipses	&	ET model (ours)	\\\\ \\hline \n'
    for label in sorted(df_iou_u['label'].unique()):
        txt_file_string +=f'{label} & '
        for index_setting, setting in enumerate(settings):
            average = np.average(all_iou[setting][label])
            n_models = len(all_iou[setting][label])
            stderror = np.std(all_iou[setting][label])*1.96/math.sqrt(n_models)
            txt_file_string += f'{average:.3f} [{average-stderror:.3f},{average+stderror:.3f}]'
            if index_setting != len(settings)-1:
                txt_file_string +=' & '
        txt_file_string +=' \\\\ \n'
    txt_file_string +=' \\hline \n \end{tabular}'
    with open(table_write_path, 'w') as f:
        f.write(txt_file_string)
    return all_iou, settings

# function used to format the calculated aucs and ious, averaged for all labels, to the format of a latex table
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
        for index_setting, setting in enumerate(settings):
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
            if index_setting!=len(settings)-1:
                txt_file_string +=' & '
        txt_file_string +=' \\\\ \n'
    txt_file_string +=' \\hline  \n \end{tabular}'
    with open('Table2.txt', 'w') as f:
        f.write(txt_file_string)

# get a table containing all the saved metrics for the validation runs
get_table(folder_runs_val)

# calculate the best thresholds for the model's output for each class, using the validation set
# thresholds are calculated separately for each setting
# Settings are separated by experiment names of the training runs
get_thresholds(folder_runs_val)#, list_methods = ['ellipse'])

# get a table containing all the saved metrics for the test runs
get_table(folder_runs_test)

# get the test aucs
get_auc(folder_runs_val)
get_auc(folder_runs_test)

# get the iou from the test set, using the best thresholds as selected above
folder_to_get_final_results = folder_runs_test
get_iou(folder_runs_val, folder_to_get_final_results, 'ellipse')
get_iou(folder_runs_val, folder_to_get_final_results, 'unet')
get_iou(folder_runs_val, folder_to_get_final_results, 'cam')

# write the latex tables to txt files
ious, settings = draw_table_latex_per_label(
    f'./iou_results_ellipse_{folder_to_get_final_results.replace("/", "|")}.csv', 
    f'./iou_results_unet_{folder_to_get_final_results.replace("/", "|")}.csv', 
    'iou', './Table4.txt')

aucs, settings = draw_table_latex_per_label(
    f'./auc_{folder_to_get_final_results.replace("/", "|")}.csv', 
    f'./auc_{folder_to_get_final_results.replace("/", "|")}.csv', 
    'auc', './Table3.txt')

draw_global_auc_iou_table_latex(ious, aucs, settings)