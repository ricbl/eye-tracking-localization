# file defining how the mimic dataset is preprocessed
import numpy as np
from .utils_dataset import TransformsDataset, SeedingPytorchTransformSeveralElements, GrayToThree, ToNumpy
from .utils_dataset import XRayResizerPadRound32, ToTensorMine, ToTensor1, get_count_positive_labels
from h5_dataset.h5_dataset import H5Dataset,change_np_type_fn
import pandas as pd
import torchvision.transforms as transforms
from .mimic_object import MIMICCXRDataset
from .global_paths import h5_path
from .list_labels import list_labels

def get_train_val_dfs():
    train_df = pd.read_csv('./train_df.csv')
    val_df = pd.read_csv('./val_df.csv')
    test_df = pd.read_csv('./test_df.csv')
    return train_df, val_df, test_df

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

post_transform_val = [ 
                        GrayToThree(),
                        ToTensorMine(),
                        normalize
                    ]

pre_transform_train = [ToTensor1(), XRayResizerPadRound32(512), transforms.Resize(512, antialias=True), ToNumpy()]

post_transform_train_with_data_aug = [
                        GrayToThree(),
                        ToTensorMine(), 
                        transforms.RandomAffine(degrees=45, translate=(0.15, 0.15),
                        scale=(0.85, 1.15), fill=0),
                        normalize
                        ]

def get_mimic_dataset_by_split(split, mimic_dataframe, post_transform, data_aug_seed):
    return TransformsDataset(SeedingPytorchTransformSeveralElements(H5Dataset(path = h5_path, 
        filename = f'{split}_joint_dataset_mimic_noseg',
        fn_create_dataset = lambda: 
            TransformsDataset(MIMICCXRDataset(mimic_dataframe), pre_transform_train, 0),
         preprocessing_functions = [change_np_type_fn(np.ubyte, 1), None], # chest x-rays are converted to 1 byte of precision to save space and disk IO when saving as hdf5 file
         postprocessing_functions = [change_np_type_fn(np.float32, 1./255.), None],
         n_processes = 16, load_to_memory = [False, True]),
            post_transform, data_aug_seed, [0]  ),[ToTensorMine()], 1  )

def get_dataset(split, data_aug_seed, use_data_aug = False, crop = False):
    train_df, val_df, test_df = get_train_val_dfs()
    if split == 'test':
        imageset = get_mimic_dataset_by_split('test', test_df, post_transform_val, data_aug_seed)
    if split == 'val':
        imageset = get_mimic_dataset_by_split('val', val_df, post_transform_val, data_aug_seed)
    if split=='train':
        imageset = get_mimic_dataset_by_split('train', train_df, post_transform_val if not use_data_aug else post_transform_train_with_data_aug, data_aug_seed)
    return imageset

if __name__=='__main__':
    # code used to print the size and number of positive labels in each split of the filtered mimic-cxr dataset
    dataset = get_dataset('train', 1, use_data_aug = False, crop = True)
    print(list_labels)
    print('Train')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_count_positive_labels(dataset)
    dataset = get_dataset('val', 1, use_data_aug = False, crop = True)
    print('Val')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_count_positive_labels(dataset)
    dataset = get_dataset('test', 1, use_data_aug = False, crop = True)
    print('Test')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_count_positive_labels(dataset)
