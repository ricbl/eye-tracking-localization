import torch
from torch.utils.data import Dataset
import numpy as np
from .utils_dataset import TransformsDataset, H5Dataset, SeedingPytorchTransformWithID, GrayToThree, ToNumpy
from .utils_dataset import XRayResizerPadRound32, return_dataloaders, ToTensorMine, ToTensor1, IteratorLoaderDifferentSizesSameBatch
import pandas as pd
import torchvision.transforms as transforms
import math
import multiprocessing
from joblib import Parallel, delayed
from .mimic_object import MIMICCXRDataset
from .eyetracking_dataset import get_dataset_et
from .global_path import h5_path

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
pre_transform_train_center_crop = [ToTensor1(), transforms.Resize(512, antialias=True), transforms.CenterCrop(512), ToNumpy()]

post_transform_train = [
                        GrayToThree(),
                        ToTensorMine(),
                        normalize
                        ]
post_transform_train_with_data_aug = [
                        GrayToThree(),
                        ToTensorMine(), 
                        transforms.RandomAffine(degrees=45, translate=(0.15, 0.15),
                        scale=(0.85, 1.15), fill=0),
                        normalize
                        ]

def get_one_sample(list_index,element_index, original_dataset_):
    print(f'{element_index}-/{len(original_dataset_[0])}')
    return original_dataset_[0][element_index]

def ait(element_index, list_index, original_dataset, list_elements):
    list_elements[list_index] = original_dataset[element_index]
    return list_elements

#dataset wrapper to load a dataset to memory for faster batch loading
class LoadToMemory(Dataset):
    def __init__(self, original_dataset, parallel = False):
        super().__init__()
        indices_iterations = np.arange(len(original_dataset))
        if parallel:
            manager = multiprocessing.Manager()
            numpys = manager.list([original_dataset])
            self.list_elements = Parallel(n_jobs=22, batch_size = 1)(delayed(get_one_sample)(list_index,element_index, numpys) for list_index, element_index in enumerate(indices_iterations))
        else:
            self.list_elements = [original_dataset[0]]*len(original_dataset)
            for list_index, element_index in enumerate(indices_iterations): 
                print(f'{element_index}+/{len(original_dataset)}')
                ait(element_index, list_index, original_dataset, self.list_elements)

    def __len__(self):
        return len(self.list_elements)
    
    def __getitem__(self, index):
        return self.list_elements[index]

def get_dataset(split, data_aug_seed, use_data_aug = False, crop = False):
    train_df, val_df, test_df = get_train_val_dfs()
    if split == 'test':
        valset = TransformsDataset(TransformsDataset(H5Dataset(lambda: LoadToMemory(
                TransformsDataset(MIMICCXRDataset(test_df), pre_transform_train_center_crop if crop else pre_transform_train)
            ,parallel=True) ,path = h5_path, filename = 'test_dataset_mimic_noseg' + ('' if not crop else '_crop'), individual_datasets = True)
            ,post_transform_val),[ToTensorMine()], 1  )
            
    if split == 'val':
        valset = TransformsDataset(TransformsDataset(H5Dataset(lambda: LoadToMemory(
                TransformsDataset(MIMICCXRDataset(val_df), pre_transform_train_center_crop if crop else pre_transform_train)
            ,parallel=True),path = h5_path, filename = 'val_dataset_mimic_noseg' + ('' if not crop else '_crop'), individual_datasets = True),
            post_transform_val),[ToTensorMine()], 1  )
            
    if split=='train':
        valset = TransformsDataset(SeedingPytorchTransformWithID(H5Dataset(lambda: LoadToMemory(
                TransformsDataset(MIMICCXRDataset(train_df), pre_transform_train_center_crop if crop else pre_transform_train)
            ,parallel=True),path = h5_path, filename = 'train_dataset_mimic_noseg' + ('' if not crop else '_crop'), individual_datasets = True),
                post_transform_train if not use_data_aug else post_transform_train_with_data_aug, data_aug_seed, [0]  ),[ToTensorMine()], 1  )
    return valset

class JoinDatasets(Dataset):
    def __init__(self, annotated_dataset, simple_dataset, use_et, grid_size):
        super().__init__()
        self.dataset_list = [simple_dataset, annotated_dataset]
        self.len_ = sum([len(self.dataset_list[i]) for i in range(len(self.dataset_list))])
        self.index_mapping = np.zeros([self.len_,2]).astype(int)
        self.use_et = use_et
        self.grid_size = grid_size
        current_index = 0
        for i in range(len(self.dataset_list)):
            self.index_mapping[current_index:current_index+len(self.dataset_list[i]),0] = i
            self.index_mapping[current_index:current_index+len(self.dataset_list[i]),1] = np.arange(len(self.dataset_list[i]))
            current_index += len(self.dataset_list[i])

    def __len__(self):
        return self.len_
    
    def __getitem__(self, index):
        if index>len(self):
            raise StopIteration
        if self.index_mapping[index,0] == 0:
            one_case = self.dataset_list[0][self.index_mapping[index,1]]
            return one_case[0], one_case[1], torch.tensor(0.),  torch.zeros(10,self.grid_size,self.grid_size) 
        if self.index_mapping[index,0] == 1:
            one_case = self.dataset_list[1][self.index_mapping[index,1]]
            return  one_case[0], one_case[1 if self.use_et else 4], torch.tensor(1.), one_case[2 if self.use_et else 3]

class JoinDatasetsRemoveAnnotations(JoinDatasets):
    def __getitem__(self, index):
        if index>len(self):
            raise StopIteration
        if self.index_mapping[index,0] == 0:
            one_case = self.dataset_list[0][self.index_mapping[index,1]]
            return one_case[0], one_case[1], torch.tensor(0.),  torch.zeros(10,self.grid_size,self.grid_size) 
        if self.index_mapping[index,0] == 1:
            one_case = self.dataset_list[1][self.index_mapping[index,1]]
            return  one_case[0], one_case[5], torch.tensor(0.),  torch.zeros(10,self.grid_size,self.grid_size) 

class Len0Dataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        raise StopIteration

class ReduceDataset(Dataset):
    def __init__(self, original_dataset, percentage_size):
        super().__init__()
        self.original_dataset = original_dataset
        self.percentage_size = percentage_size

    def __len__(self):
        return round(self.percentage_size*len(self.original_dataset))
    
    def __getitem__(self, index):
        if index>len(self):
            raise StopIteration
        return self.original_dataset[index]

class ValAnDatasets(Dataset):
    def __init__(self, annotated_dataset):
        super().__init__()
        self.annotated_dataset = annotated_dataset

    def __len__(self):
        return len(self.annotated_dataset)
    
    def __getitem__(self, index):
        if index>len(self):
            raise StopIteration
        one_case = self.annotated_dataset[index]
        return  one_case[0], one_case[4], torch.tensor(1.), one_case[3], one_case[2], one_case[5]

def get_together_dataset(split, type_, use_et, batch_size, crop, use_data_aug, num_workers, percentage_annotated, percentage_unannotated, repeat_annotated, load_to_memory,data_aug_seed, index_produce_val_image):
    original_split = split.split('_')[0]
    get_dataset_split = 'train' if original_split=='trainval' else original_split
    a = get_dataset_et(get_dataset_split, data_aug_seed, grid_size=16, use_et = use_et, use_data_aug = use_data_aug, crop = crop, load_to_memory = load_to_memory)
    u = get_dataset(get_dataset_split, data_aug_seed, use_data_aug = use_data_aug, crop = crop)
    if original_split=='train':
        a = ReduceDataset(a, percentage_annotated)
        u = ReduceDataset(u, percentage_unannotated)
    if repeat_annotated:
        assert(original_split=='train')
        assert(type_=='ua')
        a = JoinDatasets(a, Len0Dataset(), use_et = use_et, grid_size = 16)
        u = JoinDatasets(Len0Dataset(), u, use_et = use_et, grid_size = 16)
        return IteratorLoaderDifferentSizesSameBatch(
            return_dataloaders(lambda: a, original_split, batch_size = int(batch_size/2), num_workers=int(num_workers/2), index_produce_val_image = index_produce_val_image),
            return_dataloaders(lambda: u, original_split, batch_size = int(batch_size/2), num_workers=int(num_workers/2), index_produce_val_image = index_produce_val_image),
            math.ceil(len(u)/batch_size*2))
    else:
        if split.split('_')[0]=='train':
            if type_ == 'a':
                j = JoinDatasets(a, Len0Dataset(), use_et = use_et, grid_size = 16)
            elif type_=='u':
                if percentage_unannotated<1:
                    j = JoinDatasetsRemoveAnnotations(a, u, use_et = use_et, grid_size = 16)
                else:
                    j = JoinDatasets(Len0Dataset(), u, use_et = use_et, grid_size = 16)
            elif type_=='ua':                
                j = JoinDatasets(a,u, use_et = use_et, grid_size = 16)
        else:
            if split[-4:] == '_all':
                j = u
            elif split[-4:] == '_ann':
                j = ValAnDatasets(a)
        
        return return_dataloaders(lambda: j, original_split, batch_size = batch_size, num_workers=num_workers, index_produce_val_image = index_produce_val_image)

def get_count_positive_labels(dataset):
    c=[]    
    for i in range(len(dataset)):
        b = dataset[i]
        c.append(b[1])
    print(np.array(c).sum(axis=0))

if __name__=='__main__':
    dataset = get_dataset('train', 1, use_data_aug = False, crop = True)
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
    dataset = get_dataset('test', 1, use_data_aug = False, crop = True)
    print('Test')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_count_positive_labels(dataset)

    
    
    