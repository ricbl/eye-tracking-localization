import pandas as pd
from .list_labels import list_labels

def get_auc(runs_folder):
    results_list = []
    full_df = pd.read_csv('./table_summary_results.csv')
    for folder in full_df['folder'].unique():
        folder_df = full_df[full_df['folder'] == folder]
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
        for label in list_labels:
            assert(len(folder_df)==1)
            results_list.append({'folder':folder,
                                'auc':folder_df[f"score_{label.replace(' ','_')}_val_mimic_all"].values[0],
                                'setting':setting,
                                'label': label})

    pd.DataFrame(results_list).to_csv('auc.csv', index=False)
