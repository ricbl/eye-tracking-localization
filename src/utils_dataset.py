# file containing auxiliary classes or functions for pytorch datasets
from torch.utils.data import Dataset
import numpy as np
import torch
import multiprocessing
from joblib import Parallel, delayed
import torchvision.transforms as transforms
import math
import random

# get number of positive examples from each split
def get_count_positive_labels(dataset):
    all_dataset_labels = []
    for i in range(len(dataset)):
        dataset_element = dataset[i]
        # index 1 contains the image-level labels (for the eye-tracking dataset, it is the labels extracted from reports using the modified chexpert-labeler)
        all_dataset_labels.append(dataset_element[1])
    print(np.array(all_dataset_labels).sum(axis=0))

#Transforms

# Convert numpy array to float tensor while adding a channel dimension
class ToTensor1(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return torch.tensor(tensor).float().unsqueeze(0)

# Convert numpy array to tensor, keeping type
class ToTensorMine(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return torch.tensor(tensor)

# Convert pytorch tensor to numpy array, keeping type
class ToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.array(tensor)

# Triple the channel dimension of grayscale images, for use with traditional CNN that process color images
class GrayToThree(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.tile(tensor, [3,1,1])

# resize the image such that one of the dimensions is equal to size
# if fn is max, the longest dimention will be equal to size
# if fn is min, the shortest dimention will be equal to size
class XRayResizerAR(object):
    def __init__(self, size, fn):
        self.size = size
        self.fn = fn
    
    def __call__(self, img):
        old_size = img.size()[-2:]
        ratio = float(self.size)/self.fn(old_size)
        new_size = tuple([round(x*ratio) for x in old_size])
        img = transforms.Resize(new_size, antialias = True)(img)
        return img

# get the size that makes the longest dimension a multiple of size_last_layer while making the shortest dimension
# as close to size as possible
def get_32_size(shape, size, size_last_layer = 16):
    projected_max_size = size/min(np.array(shape))*max(np.array(shape))
    return round(projected_max_size/size_last_layer)*size_last_layer

# resizes an image such that its longest dimension is resized following the rules given by the get_32_size
# The shortest dimension is padded to have the same size as the longest dimension
class XRayResizerPadRound32(object):
    def __init__(self, size, size_last_layer = 16):
        self.size = size
        self.size_last_layer = size_last_layer
    
    def __call__(self, img):
        self.resizer = XRayResizerAR(get_32_size(img.size()[-2:], self.size, self.size_last_layer), max)
        img = self.resizer(img)
        pad_width = (-np.array(img.size()[-2:])+max(np.array(img.size()[-2:])))/2
        return torch.nn.functional.pad(img, (math.floor(pad_width[1]),math.ceil(pad_width[1]),math.floor(pad_width[0]),math.ceil(pad_width[0])))

# used for resizing the eyetracking heatmaps and ellipse labels to the size of the output grid of the network.
# Max pooling is used to guarantee that there will be at least one positive grid cell from each eye-tracking heatmap 
# after thresholding it.
class ResizeHeatmap(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
    
    def __call__(self, masks_labels):
        return torch.nn.AdaptiveMaxPool2d((self.grid_size,self.grid_size))(masks_labels)

# Dataset/dataloader Wrappers

# dataset set wrapper that concatenates the datasets listed in dataset_list
class JoinDatasets(Dataset):
    def __init__(self, dataset_list):
        super().__init__()
        self.dataset_list = dataset_list
        self.len_ = sum([len(self.dataset_list[i]) for i in range(len(self.dataset_list))])
        self.index_mapping = np.zeros([self.len_,2]).astype(int)
        current_index = 0
        for i in range(len(self.dataset_list)):
            self.index_mapping[current_index:current_index+len(self.dataset_list[i]),0] = i
            self.index_mapping[current_index:current_index+len(self.dataset_list[i]),1] = np.arange(len(self.dataset_list[i]))
            current_index += len(self.dataset_list[i])
    
    def __len__(self):
        return self.len_
    
    def __getitem__(self, index):
        return self.dataset_list[self.index_mapping[index,0]][self.index_mapping[index,1]]

#dataset wrapper to load a dataset to memory for faster batch loading.
# The loading can be done using several threads by setting positive n_processes
class LoadToMemory(Dataset):
    def __init__(self, original_dataset, n_processes = 0):
        super().__init__()
        indices_iterations = np.arange(len(original_dataset))
        if n_processes>0:
            manager = multiprocessing.Manager()
            numpys = manager.list([original_dataset])
            self.list_elements = Parallel(n_jobs=n_processes, batch_size = 1)(delayed(self.get_one_sample)(list_index,element_index, numpys) for list_index, element_index in enumerate(indices_iterations))
        else:
            self.list_elements = [original_dataset[0]]*len(original_dataset)
            for list_index, element_index in enumerate(indices_iterations): 
                print(f'{element_index}+/{len(original_dataset)}')
                self.list_elements[list_index] = original_dataset[element_index]
                self.ait(element_index, list_index, original_dataset, self.list_elements)
    
    def __len__(self):
        return len(self.list_elements)
    
    def __getitem__(self, index):
        return self.list_elements[index]
    
    def get_one_sample(self, list_index,element_index, original_dataset_):
        print(f'{element_index}-/{len(original_dataset_[0])}')
        return original_dataset_[0][element_index]

# Class used to join two dataloaders of different sizes, reurning one batch from each loader in
# each iteration. When the any loader iterates through all its elements, it is restarted, until n_iter_per_epoch iterations
# are performed.
class IteratorLoaderDifferentSizesSameBatch:
    def __init__(self, loaders, n_iter_per_epoch):
        self.loaders = loaders
        self.__iter__()
        self.count_iterations = 0
        self.n_iter_per_epoch = n_iter_per_epoch
    
    def __iter__(self):
        self.iterLoaders = [iter(loader) for loader in self.loaders]
        return self
    
    def nextI(self, this_iter):
        return next(this_iter,None)
    
    def __next__(self):
        if self.count_iterations >= self.n_iter_per_epoch:
            self.count_iterations = 0
            raise StopIteration
        current_batch_loader = []
        for i in range(len(self.loaders)):
            current_batch_loader.append(self.nextI(self.iterLoaders[i]))
            if current_batch_loader[i] is None:
                self.iterLoaders[i] = iter(self.loaders[i])
                current_batch_loader[i] = self.nextI(self.iterLoaders[i])
        self.count_iterations += 1
        to_return = []
        for index_in_batch in range(len(current_batch_loader[0])):
            current_tensor = []
            for sublist in current_batch_loader:
                current_tensor.append(sublist[index_in_batch])
            to_return.append(torch.cat(current_tensor))
        return to_return
      
    next = __next__

#dataset wrapper to apply transformations to a pytorch dataset. indices_to_transform defines the indices of the elements
# of the tuple returned by original_dataset to which the transformation should be applied
class TransformsDataset(Dataset):
    def __init__(self, original_dataset, original_transform, indices_to_transform):
        if isinstance(indices_to_transform, int):
            indices_to_transform = [indices_to_transform]
        self.original_transform = original_transform
        if type(self.original_transform)==type([]):
            self.original_transform  = transforms.Compose(self.original_transform)
        self.indices_to_transform = indices_to_transform
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        to_return = self.original_dataset[index]
        fn_run_transform = (lambda s,a: s.original_transform(a))
        for i in self.indices_to_transform:
            to_return = [*to_return[:i], fn_run_transform(self, to_return[i]), *to_return[(i+1):]]
        return to_return

# Dataset wrapper that applies a sequence of transformations original_transform to elements indexed by 
# indices_to_transform while giving the same random seed to all transformations for the same sample
class SeedingPytorchTransformSeveralElements(TransformsDataset):
    def __init__(self, original_dataset, original_transform, seed, indices_to_transform):
        super().__init__(original_dataset, original_transform, indices_to_transform)
        self.seed = seed
        self.previouspythonstate = None
        self.previoustorchstate = None
    
    def __getitem__(self, index):
        x = self.original_dataset[index]
        fn_run_transform = (lambda s,a: s.original_transform(a))
        
        #saving the random state from outside to restore it afterward
        outsidepythonstate = random.getstate()
        outsidetorchstate = torch.random.get_rng_state()
        
        to_return = x
        for i in self.indices_to_transform:
            if self.previouspythonstate is None:
                if self.seed is not None:
                    # for the first sample in a training, set a seed
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
            else:
                # restores the state from either last sample or last element of this sample
                random.setstate(self.previouspythonstate)
                torch.random.set_rng_state(self.previoustorchstate)
            
            # saves data augmentation random state to use the same state for all elements of this sample
            self.previouspythonstate = random.getstate() 
            self.previoustorchstate = torch.random.get_rng_state()
            
            # apply transform to element i
            to_return = [*to_return[:i], fn_run_transform(self, to_return[i]), *to_return[(i+1):]]
        
        # saves data augmentation random state to continue from same state when next sample is sampled
        self.previouspythonstate = random.getstate() 
        self.previoustorchstate = torch.random.get_rng_state()
        
        #restoring external random state
        random.setstate(outsidepythonstate)
        torch.random.set_rng_state(outsidetorchstate)
        
        return to_return

# Reduces a dataset to a subset of itself, where the subset is given by a list of indices over the original dataset
class ChangeDatasetToIndexList(Dataset):
    def __init__(self, original_dataset, index_list):
        self.original_dataset = original_dataset
        self.index_list = index_list
    
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        return self.original_dataset[self.index_list[index]]

#generic function to get dataloaders from datasets
def return_dataloaders(instantiate_dataset, split, batch_size, num_workers, index_produce_val_image):
    #if the dataset should only contain specific images (for image generation for the paper)
    if len(index_produce_val_image)>0 and split!='train':
        instantiate_dataset_ = instantiate_dataset
        instantiate_dataset = lambda :ChangeDatasetToIndexList(instantiate_dataset_(), index_produce_val_image)
    
    return torch.utils.data.DataLoader(dataset=instantiate_dataset(), batch_size=batch_size,
                        shuffle=(split=='train'), num_workers=num_workers, pin_memory=True, drop_last = (split=='train'))