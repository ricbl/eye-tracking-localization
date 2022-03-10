import pandas as pd
import os

def get_iou(val_runs_folder, test_runs_folder):
    df_thresholds = pd.read_csv('./best_thresholds.csv')
    table_summary = pd.read_csv('./table_summary_results.csv')
    df_thresholds = df_thresholds[df_thresholds['method']=='cam']
    test_folders = {}
    folders_runs = []
    for _, dirnames, _ in os.walk(test_runs_folder):
        folders_runs += dirnames
        break
    for index_folder, folder in enumerate(folders_runs):
        with open(f'{test_runs_folder}/{folder}/log.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.split(':')[2]=='load_checkpoint_d':
                    load_checkpoint_d = line.split(':')[3][1:]
        original_id = int(load_checkpoint_d.split('/')[-2][-4:])
        test_folders[original_id] = folder

    iou_results = []
    for index, row in df_thresholds.iterrows():
        folder = row['folder']
        with open(f'{val_runs_folder}/{folder}/log.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.split(':')[2]=='load_checkpoint_d':
                    load_checkpoint_d = line.split(':')[3][1:]
        original_id = int(load_checkpoint_d.split('/')[-2][-4:])
        
        label = row['label']
        setting = row['setting']
        threshold = row['best_threshold']    
        row_test = table_summary[table_summary['folder']==test_folders[original_id]]
        assert(len(row_test)==1)
        iou = row_test[f'iou_{label.replace(" ","_")}_val_cam_iou_{threshold}'].values[0]
        iou_results.append({'original_id':original_id,'folder':folder,'setting':setting,'label':label,'threshold':threshold,'iou':iou})
        
    pd.DataFrame(iou_results).to_csv('iou_results.csv', index=False)