import torchvision.transforms as transforms
import torch
from .utils_dataset import TransformsDataset, H5Dataset, LoadToMemory, SeedingPytorchTransformWithID, GrayToThree, DiscretizeET, ToTensorMine, ToTensor1, XRayResizerPadRound32, ToNumpy
from .eyetracking_object import ETDataset
import pandas as pd
import numpy as np
from .global_paths import h5_path
from .list_labels import list_labels

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

pre_transform_train = [ToTensor1(), XRayResizerPadRound32(512), transforms.Resize(512, antialias=True), ToNumpy()]
pre_transform_train_center_crop = [ToTensor1(), transforms.Resize(512, antialias=True), transforms.CenterCrop(512), ToNumpy()]

post_transform_train = [
                        transforms.RandomAffine(degrees=45, translate=(0.15, 0.15),
                        scale=(0.85, 1.15), fill=0),
                        ]

def get_dataset_et(split, data_aug_seed, grid_size=8, use_et = True, use_data_aug = False, crop = False, load_to_memory = False):
    if split == 'test':
        valset = TransformsDataset(TransformsDataset(SeedingPytorchTransformWithID(SeedingPytorchTransformWithID(TransformsDataset(SeedingPytorchTransformWithID(H5Dataset(lambda: 
        LoadToMemory(
                ETDataset('test',3, pre_transform_train if not crop else pre_transform_train_center_crop) , parallel=True)
            ,path = h5_path, filename = 'test_dataset_et_3_noseg' + ('' if not crop else '_crop'), individual_datasets = True), 
            [], 0, [0,2,3]  ),
            [GrayToThree(), ToTensorMine()]),
            [ToTensorMine()], 0, [1,2,3,4,5]  ), 
            [], 0, [0,2,3]  ),
            [normalize]), [DiscretizeET(grid_size)], 2)
    if split == 'val':
        valset = TransformsDataset(TransformsDataset(SeedingPytorchTransformWithID(SeedingPytorchTransformWithID(TransformsDataset(SeedingPytorchTransformWithID(H5Dataset(lambda: 
                LoadToMemory(ETDataset('val',3, pre_transform_train if not crop else pre_transform_train_center_crop), parallel=True)
            ,path = h5_path, filename = 'val_dataset_et_3_noseg' + ('' if not crop else '_crop'), individual_datasets = True),
            [], 0, [0,2,3]  ),
            [GrayToThree(),ToTensorMine()]),
            [ToTensorMine()], 0, [1,2,3,4,5]  ), 
            [], 0, [0,2,3]  ),
            [normalize]), [DiscretizeET(grid_size)], 2)
    if split == 'train':
        valset = SeedingPytorchTransformWithID(TransformsDataset(H5Dataset(lambda:
                LoadToMemory(ETDataset('train',3, pre_transform_train if not crop else pre_transform_train_center_crop), parallel=True)
            ,path = h5_path, filename = 'train_dataset_et_3_noseg' + ('' if not crop else '_crop'), individual_datasets = True), 
            [GrayToThree(), ToTensorMine()]),
                 [ToTensorMine()], 0, [1,2,3,4,5]  )
        if not use_data_aug:
            valset = TransformsDataset(TransformsDataset(valset, [normalize]), [DiscretizeET(grid_size)], 2)
            if not use_et:
                valset = TransformsDataset(valset, [DiscretizeET(grid_size)], 3)
    if load_to_memory:
        valset = LoadToMemory(valset, parallel=(split == 'train') )
    if split == 'train':
        if use_data_aug:
            valset = TransformsDataset(TransformsDataset(SeedingPytorchTransformWithID(valset,
            post_transform_train, data_aug_seed, [0,2,3]  ), [normalize]), [DiscretizeET(grid_size)], 2)
            if not use_et:
                valset = TransformsDataset(valset, [DiscretizeET(grid_size)], 3)
    return valset

def get_list_of_values_from_grid_for_histogram():
    c = []
    for phase in [3]:
        a = H5Dataset(lambda: 
        LoadToMemory(
                ETDataset('train',3, pre_transform_train) , parallel=True)
            ,path = h5_path, filename = 'train_dataset_et_3_noseg', individual_datasets = True)
        for i in range(len(a)):
            d = a[i]
            e = DiscretizeET(16)(torch.tensor(d[2])).numpy().reshape([-1])
            for j in range(len(e)):
                if e[j]>0.005:
                    b = {}
                    b['grid_cell'] = e[j]
                    c.append(b)
            print(i)
    pd.DataFrame(c).to_csv('test_histogram_grid.csv')

# get number of positive examples from each split
def get_count_positive_labels(dataset):
    c=[]
    for i in range(len(dataset)):
        b = dataset[i]
        c.append(b[1])
    print(np.array(c).sum(axis=0))

if __name__=='__main__':
    get_list_of_values_from_grid_for_histogram()
    print(list_labels)
    dataset = get_dataset_et('train', 1, grid_size=16, use_et = True, use_data_aug = True, crop = False)
    print('Train')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_count_positive_labels(dataset)
    dataset = get_dataset_et('val', 1, grid_size=16, use_et = True, use_data_aug = True, crop = False)
    print('Val')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_count_positive_labels(dataset)
    dataset = get_dataset_et('test', 1, grid_size=16, use_et = True, use_data_aug = True, crop = False)
    print('Test')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_count_positive_labels(dataset)
    
