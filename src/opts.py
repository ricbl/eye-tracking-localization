# script used to define inputs user can add to the command line calling of the training script, and a few other settings
import argparse
import time
from random import randint
import os
import socket
import numpy as np
thresholds = [0,0.001,0.01, 0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95, 0.975, 0.99, 0.999]

#convert a few possibilities of ways of inputing boolean values to a python boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_opt():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--save_folder', type=str, nargs='?', default='./runs',
                                help='If you want to save files and outputs in a folder other than \'./runs\', change this variable.')
    parser.add_argument('--gpus', type=str, nargs='?', default=None,
                                help='Set the gpus to use, using CUDA_VISIBLE_DEVICES syntax.')
    parser.add_argument('--experiment', type=str, nargs='?', default='',
                                help='Set the name of the folder where to save the run.')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0,
                                help='Weight decay of the optimizer.')
    parser.add_argument('--dataset_type', type=str, nargs='?', default='ua',
                                help='a for annotated, u for unnanotated, and ua for both')
    parser.add_argument('--skip_train', type=str2bool, nargs='?', default='false',
                                help='If you just want to run validation, set this value to true.')
    parser.add_argument('--skip_validation', type=str2bool, nargs='?', default='false',
                                help='If True, no validation is performed.')
    parser.add_argument('--split_validation', type=str, nargs='?', default='val',
                                    help='Use \'val\' to use the validation set for calculating scores every epoch. Use \'test\' for using the test set during scoring.')
    parser.add_argument('--load_checkpoint_d', type=str, nargs='?', default=None,
                                    help='If you want to start from a previously trained classifier, set a filepath locating a model checkpoint that you want to load')
    parser.add_argument('--nepochs', type=int, nargs='?', default=60,
                                help='Number of epochs to run training and validation')
    parser.add_argument('--loss', type=str, nargs='?', default='li',
                                help='Changes the type of loss for training. The options are: ce, or li')
    parser.add_argument('--use_et', type=str2bool, nargs='?', default='true',
                                help='If False, the ellipses ground truths are used as localiation annotation for training, instead of the eye-tracking data.')
    parser.add_argument('--threshold_box_label', type=float, nargs='*', default=[0.15],
                                help='The threshold used in the eye-tracking heatmap used as dataset annotation.')
    parser.add_argument('--use_lr_scheduler', type=str2bool, nargs='?', default='false',
                                help='If True, uses a learning rate scheduler that reduces learning rate over training.')
    parser.add_argument('--use_center_crop', type=str2bool, nargs='?', default='false',
                                help='If True, the images are center cropped instead of padded to a square image.')
    parser.add_argument('--use_pretrained', type=str2bool, nargs='?', default='true',
                                help='If True, the resnet model is loaded with PyTorch ImagNet-trained weights before start of training.')
    parser.add_argument('--use_data_augmentation', type=str2bool, nargs='?', default='true',
                                help='If True, data augmentation is used during training.')
    parser.add_argument('--optimizer', type=str, nargs='?', default='adamams',
                                help='adamw or adamams, to choos the tye of optimizer.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=20,
                                help='Batch size for training and validation.')
    parser.add_argument('--num_workers', type=int, nargs='?', default=5,
                                help='Number of threads created for faster data loading.')
    parser.add_argument('--percentage_annotated', type=float, nargs='?', default=1.,
                                help='Number from 0 to 1 limiting the number of images from the annotated dataset.')
    parser.add_argument('--percentage_unannotated', type=float, nargs='?', default=1.,
                                help='Number from 0 to 1 limiting the number of images from the unannotated dataset.')
    parser.add_argument('--repeat_annotated', type=str2bool, nargs='?', default='false',
                                help='If True, creates batches for training with the same number of annotated and unannotated images. Annotated images are repeated in the same epoch since there are less annotated image than unannotated.')
    parser.add_argument('--weight_loss_annotated', type=float, nargs='?', default=5.0,
                                help='Hyperparametr \lambda_A from Equation 5 in the paper. It controls the relative importance of annotated images during training')
    parser.add_argument('--load_to_memory', type=str2bool, nargs='?', default='false',
                                help='If True, loads datasets to RAM memory for faster data loading.')
    parser.add_argument('--data_aug_seed', type=int, nargs='?', default=None,
                                help='Sets a seed for the randomness inthe data augmentation')
    parser.add_argument('--threshold_ior', type=str2bool, nargs='?', default='false',
                                help='If True, validation IoU numbers are calculated in terms of percentage of IoU larger than 0.1.')
    parser.add_argument('--calculate_cam', type=str2bool, nargs='?', default='false',
                                help='If True, validation includes localization IoU numbers using the gradCAM algorithm.')
    parser.add_argument('--validate_iou', type=str2bool, nargs='?', default='true',
                                help='If True, validation includes localization IoU numbers.')
    parser.add_argument('--validate_auc', type=str2bool, nargs='?', default='true',
                                help='If True, validation includes image-level label AUC numbers.')
    parser.add_argument('--index_produce_val_image', type=int, nargs='*', default=[],
                            help='Sets a list of image indices that will be used to limit the images loaded for the validation/test set.')
    parser.add_argument('--draw_images', type=str2bool, nargs='?', default='false',
                                help='If True, draws images for the paper.')
    parser.add_argument('--label_to_draw', type=int, nargs='?', default=None,
                            help='Used to choose which image-level label is draw when drawing images.')
    parser.add_argument('--sm_suffix', type=str, nargs='?', default=None,
                            help='Sets a suffix to put at the end of filenames for draw_images.')
    parser.add_argument('--use_grid_balancing_loss', type=str2bool, nargs='?', default='false',
                            help='If true, the normalization of the Li et al. loss is modified to depend on the number of grid celss that are being multiplied.')
    parser.add_argument('--grid_size', type=int, nargs='?', default=16,
                            help='The size used to resize eye-tracking heatmaps and ellipses when used as annotations for training the model.')
    parser.add_argument('--last_layer_index', type=int, nargs='*', default=[4],
                            help='A list of the spatial layers from resnet to include as output of the model. Number from 1 to 4 are allowed, where 4 is the last spatial layer.')
    parser.add_argument('--use_mixed_precision', type=str2bool, nargs='?', default='false',
                            help='If true, pytorch\'s automatic mixed precison is used for training')
    parser.add_argument('--dataset_type_et', type=str, nargs='?', default='h5',
                            help='Type of file to save preprocessed dataset to. Options: h5, zarr, png, mmap')
    parser.add_argument('--sequence_model', type=str2bool, nargs='?', default='false',
                            help='If true, use the heatmaps of every sentence as input to a sequence model. Not used in the paper.')
    parser.add_argument('--dataset_name', type=str, nargs='?', default='xray',
                            help='Only used if sequence_model is true. Accepted values: "xray" or "toy". Toy is a dataset with synthetic images using spread digits from MNIST and randomly selected sequence of fixations.')
    parser.add_argument('--checkpoint_sequence', type=str, nargs='?', default=None,
                            help='Checkpoint for the sequence model.')
    parser.add_argument('--type_toy', type=str, nargs='?', default='original',
                            help='Changes details of how the toy dataset is generated. Accepted values: "original", "no_negatives", "fixed_sentences", or "inverted"')
    parser.add_argument('--unet', type=str2bool, nargs='?', default='false',
                            help='If true, use the multitask learning from Creation and validation of a chest x-ray dataset with eye-tracking and report dictation for AI development, Karargyris  et al..')
    parser.add_argument('--gamma_unet', type=float, nargs='?', default=1,
                            help='Weight given to the multitask/decoder part of the loss.')
    parser.add_argument('--sigma_loss_multiplier', type=float, nargs='?', default=0,
                            help='If a weight higher than 0 is used, the sigma uncertainty loss (Follow My Eye: Using Gaze to Supervise Computer-Aided Diagnosis, Wang et al.) will be activated. No result was presented in the paper with this loss because no improvement was seen.')
    parser.add_argument('--enforce_negatives_sigma', type=str2bool, nargs='?', default='false',
                            help='If false, enforce the GradCAM heatmap to be similar to the label-specific eye-tracking heatmaps for positive labels. If true, in addition to those cases, also enforce the heatmaps from gradcam to be fully zeros when a label is not present in the image.')
    parser.add_argument('--calculate_label_specific_heatmaps', type=str2bool, nargs='?', default='true',
                            help='If false, use the heatmaps of the whole report/dictation as annotation, using a single heatmap for every single class label. Only used for ablation study.')
    args = parser.parse_args()
    
    # if only one threshold for the eye-tracking heatmaps was given, make it the same threshold for all labels
    if len(args.threshold_box_label)==1:
        args.threshold_box_label = args.threshold_box_label*10
    
    args.thresholds_iou = thresholds
    if args.skip_validation:
        args.metric_to_validate = 'auc_average_train' # if no valiadtion is performed, use the training AUC for deciding the best epoch
    else:
        args.metric_to_validate = 'auc_average_val_mimic_all' # if valiadtion is performed, use the validation AUC for deciding the best epoch
    args.function_to_compare_validation_metric = lambda x,y:x>=y #the larger the metric the better
    args.initialization_comparison = float('-inf')

    #gets the current time of the run, and adds a four digit number for getting
    #different folder name for experiments run at the exact same time.
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + str(randint(1000,9999))
    args.timestamp = timestamp
    
    #register a few values that might be important for reproducibility
    args.screen_name = os.getenv('STY')
    args.hostname = socket.gethostname()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES']
    import platform
    args.python_version = platform.python_version()
    import torch
    args.pytorch_version = torch.__version__ 
    import torchvision
    args.torchvision_version = torchvision.__version__
    args.numpy_version = np.__version__
    return args
