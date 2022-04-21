# classes used to put the unannotated and annotated datasets into the same dataset/dataloader
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils_dataset import return_dataloaders, IteratorLoaderDifferentSizesSameBatch
import math
from .eyetracking_dataset import get_dataset_et
from .mimic_dataset import get_dataset as get_dataset_mimic

# class used to put annotated dataset and unannotated dataset in the same dataset.
# The outputs of the __getitem__ function have to be standardized in content, indexing and size, leading to a return with 4 elements:
# element 0: chest x-ray image
# element 1: image-level labels
# element 2: values representing if a location annotation is present (0 for absent, 1 for present)
# element 3: location annotation (all zeros if absent). It might be sourced from eye-tracking data or ellipses drawn by radiologists
class JoinDatasetsAnotatedUnannotated(Dataset):
    def __init__(self, annotated_dataset, simple_dataset, use_et, grid_size):
        super().__init__()
        self.dataset_list = [simple_dataset, annotated_dataset]
        self.len_ = sum([len(self.dataset_list[i]) for i in range(len(self.dataset_list))])
        self.index_mapping = np.zeros([self.len_,2]).astype(int)
        self.use_et = use_et
        self.grid_size = grid_size
        current_index = 0
        
        # create a mapping from the index of an example of the joint dataset to the dataset it belongs to
        # and to the index that should be used to get that example from the original dataset
        for dataset_index in range(len(self.dataset_list)):
            self.index_mapping[current_index:current_index+len(self.dataset_list[dataset_index]),0] = dataset_index # index 0 contains the dataset identifier
            self.index_mapping[current_index:current_index+len(self.dataset_list[dataset_index]),1] = np.arange(len(self.dataset_list[dataset_index])) # index 1 contains the index for the __getitem__ function of the original dataset
            current_index += len(self.dataset_list[dataset_index])

    def __len__(self):
        return self.len_
    
    def __getitem__(self, index):
        if index>len(self):
            raise StopIteration
        # if image indexed with index is from the unannotated dataset
        if self.index_mapping[index,0] == 0:
            one_case = self.dataset_list[0][self.index_mapping[index,1]]
            # element 0: chest x-ray image
            # element 1:
            # image-level labels from the MIMIC-CXR dataset
            # element 2:
            # return 0 to indicate that this image does not contains annotation
            # element 3:
            # return an array of zeros with the same shape of the ellipse annotations
            return one_case[0], one_case[1], torch.tensor(0.),  torch.zeros(10,self.grid_size,self.grid_size) 
        # if image indexed with index is from the annotated dataset
        if self.index_mapping[index,0] == 1:
            one_case = self.dataset_list[1][self.index_mapping[index,1]]
            # element 0: chest x-ray image
            # element 1:
            # if self.use_et is True, return the image-level labels extracted from reports as ground-truth (index 1)
            # if self.use_et is False, use image-level labels as given by radiologists as ground-truth (index 4)
            # element 3:
            # return 1 to indicate that this image contains annotation
            # element 4:
            # if self.use_et is True, return the eye-tracking heatmap as annotation ground-truth (index 2)
            # if self.use_et is False, use ellipses as annotation ground-truth (index 3)
            return  one_case[0], one_case[1 if self.use_et else 4], torch.tensor(1.), one_case[2 if self.use_et else 3]

# dataset used to join annotated and unannotated datasets while transforming
#  examples from the annotated dataset into unannotated cases
class JoinDatasetsRemoveAnnotations(JoinDatasetsAnotatedUnannotated):
    def __getitem__(self, index):
        if index>len(self):
            raise StopIteration
        if self.index_mapping[index,0] == 0:
            one_case = self.dataset_list[0][self.index_mapping[index,1]]
            # element 0: chest x-ray image
            # element 1:
            # image-level labels from the MIMIC-CXR dataset
            return one_case[0], one_case[1], torch.tensor(0.),  torch.zeros(10,self.grid_size,self.grid_size) 
        if self.index_mapping[index,0] == 1:
            one_case = self.dataset_list[1][self.index_mapping[index,1]]
            # element 0: chest x-ray image
            # element 1: labels from the mimic-cxr dataset
            # element 3: return 0 for "unannotated", since annotation has been removed
            # element 4: return all zeros as annotation
            return  one_case[0], one_case[5], torch.tensor(0.),  torch.zeros(10,self.grid_size,self.grid_size) 

# dataset with no examples (length 0)
class Len0Dataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        raise StopIteration

# dataset used to remove a percentage of samples from the end of a dataset
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

#changing the order of elements for the annotated validation dataset
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
        # element 0: chest x-ray
        # element 1: image-level labels as given by radiologists
        # element 2: 1 to represent the fact that the image is annotated
        # element 3: ellipse annotations
        # element 4: eye-tracking heatmaps
        # element 5: mimic-cxr image-level labels
        return  one_case[0], one_case[4], torch.tensor(1.), one_case[3], one_case[2], one_case[5]

# function to get a dataset that includes unannotated images and anottated images in the same dataset
def get_together_dataset(split, type_, use_et, batch_size, crop, use_data_aug, num_workers, percentage_annotated, percentage_unannotated, repeat_annotated, load_to_memory,data_aug_seed, index_produce_val_image, grid_size, dataset_type_et):
    original_split = split.split('_')[0]
    
    # 'trainval' is used to pass the training set through the same calculations as the validation set
    # In that case, get_dataset_split=='train' and original_split=='val'
    get_dataset_split = 'train' if original_split=='trainval' else original_split
    
    annotated_part = get_dataset_et(get_dataset_split, data_aug_seed, grid_size=grid_size, use_et = use_et, use_data_aug = use_data_aug, crop = crop, load_to_memory = load_to_memory, dataset_type = dataset_type_et)
    unannotated_part = get_dataset_mimic(get_dataset_split, data_aug_seed, use_data_aug = use_data_aug, crop = crop)
    if original_split=='train':
        annotated_part = ReduceDataset(annotated_part, percentage_annotated)
        unannotated_part = ReduceDataset(unannotated_part, percentage_unannotated)
        print(len(annotated_part))
        print(len(unannotated_part))
        
    # if repeat annotated is True, the datasets, instead of concatenated, are sampled independently
    # with the length of the unannotated dataset dictating the length of an epoch, and the unannotated dataset
    # repeating several times in the same epoch. The number of samples from each of these two datasets is the same
    # for each epoch
    if repeat_annotated:
        assert(original_split=='train')
        assert(type_=='ua')
        annotated_part = JoinDatasetsAnotatedUnannotated(annotated_part, Len0Dataset(), use_et = use_et, grid_size = grid_size)
        unannotated_part = JoinDatasetsAnotatedUnannotated(Len0Dataset(), unannotated_part, use_et = use_et, grid_size = grid_size)
        return IteratorLoaderDifferentSizesSameBatch(
            [return_dataloaders(lambda: annotated_part, original_split, batch_size = int(batch_size/2), num_workers=int(num_workers/2), index_produce_val_image = index_produce_val_image),
            return_dataloaders(lambda: unannotated_part, original_split, batch_size = int(batch_size/2), num_workers=int(num_workers/2), index_produce_val_image = index_produce_val_image)],
            math.ceil(len(unannotated_part)/batch_size*2))
    else:
        if split.split('_')[0]=='train':
            if type_ == 'a': # if use only the annotated dataset
                j = JoinDatasetsAnotatedUnannotated(annotated_part, Len0Dataset(), use_et = use_et, grid_size = grid_size)
            elif type_=='u': # if use only the unannotated dataset
                if percentage_unannotated<1: # if not using the full unannotated dataset, include the annotated images as unannotated
                    j = JoinDatasetsRemoveAnnotations(annotated_part, unannotated_part, use_et = use_et, grid_size = grid_size)
                else:
                    j = JoinDatasetsAnotatedUnannotated(Len0Dataset(), unannotated_part, use_et = use_et, grid_size = grid_size)
            elif type_=='ua': # if use both dataset, join them
                j = JoinDatasetsAnotatedUnannotated(annotated_part,unannotated_part, use_et = use_et, grid_size = grid_size)
        else:
            #for 'test_all' or 'val_all', return the full validation mimic-cxr dataset
            if split[-4:] == '_all':
                j = unannotated_part
            #for 'test_ann' or 'val_ann', return the annotated dataset, with the order of elements changed
            elif split[-4:] == '_ann':
                j = ValAnDatasets(annotated_part)
        
        return return_dataloaders(lambda: j, original_split, batch_size = batch_size, num_workers=num_workers, index_produce_val_image = index_produce_val_image)
