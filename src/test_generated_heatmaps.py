# script used to test which type of sentence heatmap gets a higher iou against ellipses for the associated abnormalities.
# Very similar to testing the models, but loading heatmaps from sentences instead of from models
# This validation is performed using cases from Phase 1 and Phase 2, which do not intersect with the cases from Phase 3 used in the paper
from argparse import Namespace
from .eyetracking_object import ETDataset
from .utils_dataset import JoinDatasets, H5Dataset
from . import output as outputs
from . import metrics
from .table_summary_results import get_table
from .iou_results import get_iou
from .get_best_threshold_per_label import get_thresholds
from .train import localization_score_fn
import pathlib
import os
import torch
from .find_fixations_all_sentences import create_heatmaps
from .global_paths import  eyetracking_dataset_path
from .eyetracking_dataset import get_h5_dataset
from .eyetracking_dataset import pre_transform_train
thresholds_iou = [0,0.001,0.01, 0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95, 0.975, 0.99, 0.999]
folder_runs_val = 'test_extracting_strategies/'

pathlib.Path(folder_runs_val).mkdir(parents=True, exist_ok=True) 

#generate the sentence heatmaps for phases 1 and 2 for a specific method
def get_heatmaps(heatmap_method):
    preprocessed_heatmaps_location = f'./{heatmap_method}_heatmaps/'
    file_phase_1 = 'metadata_phase_1.csv'
    create_heatmaps(1, eyetracking_dataset_path, file_phase_1,
                      folder_name=f'{preprocessed_heatmaps_location}/heatmaps_sentences_phase_1', method = heatmap_method)
    file_phase_2 = 'metadata_phase_2.csv'
    create_heatmaps(2, eyetracking_dataset_path, file_phase_2,
                      folder_name=f'{preprocessed_heatmaps_location}/heatmaps_sentences_phase_2', method = heatmap_method)

def run_one_batch(element, metric):
    boxes = torch.tensor(element[3])
    heatmap = torch.tensor(element[2])
    label = torch.tensor(element[4])
    
    for threshold in thresholds_iou:
        iou = localization_score_fn(boxes, heatmap,threshold)
        metric.add_iou(f'val_ellipse_iou_{threshold}', iou, label)

def run_epoch(heatmap_method,etdataset_loader,index_heatmap_method):
    experiment = f'{heatmap_method}'
    
    if not os.path.exists(folder_runs_val + '/' + experiment):
        # create a folder for the experiment
        pathlib.Path(folder_runs_val + '/' + experiment).mkdir(parents=True, exist_ok=True) 
        opt = Namespace(**{'load_checkpoint_d':f'./{folder_runs_val}/{experiment}/000{index_heatmap_method}', 
            'percentage_annotated':1.,
            'use_et':True,
            'dataset_type':f'{experiment}'})
        output = outputs.Outputs(opt, folder_runs_val + '/' + experiment)
        output.save_run_state(os.path.dirname(__file__))
        metric = metrics.Metrics(False, False)
        
        # iterate through the whole dataset, calculating the iou for each example
        for index_element, element in enumerate(etdataset_loader):
            print(index_element)
            run_one_batch(element, metric)
        
        # write the resulting iou to the log file
        output.log_added_values(0, metric) 

def run_thresholds_one_method(heatmap_method, index_heatmap_method):
    preprocessed_heatmaps_location = f'./{heatmap_method}_heatmaps/'
    
    # generate an eye-tracking dataset using the sentence heatmaps extracted with the respective method
    etdataset = JoinDatasets([ETDataset('train', 1,pre_transform_train, preprocessed_heatmaps_location),ETDataset('train', 2, pre_transform_train, preprocessed_heatmaps_location)])
    etdataset = get_h5_dataset(f'{heatmap_method}', [False, False, True, True, False, False], H5Dataset, False, lambda: etdataset)
    etdataset_loader = torch.utils.data.DataLoader(dataset=etdataset, batch_size=20,
                        shuffle=False, num_workers=5, pin_memory=True, drop_last = False)
    run_epoch(heatmap_method,etdataset_loader, index_heatmap_method)

if __name__=='__main__':
    for index_heatmap_method, heatmap_method in enumerate(['5s','single','double','all','allsentences', 'andpause', '2.5s', '7.5s']):
        get_heatmaps(heatmap_method)
        run_thresholds_one_method(heatmap_method, index_heatmap_method)

    get_table(folder_runs_val)
    get_thresholds(folder_runs_val, ['ellipse'])
    get_iou(folder_runs_val, folder_runs_val, 'ellipse')

