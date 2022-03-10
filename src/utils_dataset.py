from torch.utils.data import Dataset
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import os

import signal
import hashlib
import shutil
import torch
import h5py
import pathlib
import multiprocessing
from joblib import Parallel, delayed
import torchvision.transforms as transforms
import math
import PIL
import random

class ToTensor1(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return torch.tensor(tensor).float().unsqueeze(0)

class ToTensorMine(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return torch.tensor(tensor)
        
class ToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.array(tensor)

class GrayToThree(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.tile(tensor, [3,1,1])

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

def get_32_size(shape, size):
    projected_max_size = size/min(np.array(shape))*max(np.array(shape))
    return round(projected_max_size/16)*16

class XRayResizerPadRound32(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        self.resizer = XRayResizerAR(get_32_size(img.size()[-2:], self.size), max)
        img = self.resizer(img)
        pad_width = (-np.array(img.size()[-2:])+max(np.array(img.size()[-2:])))/2
        return torch.nn.functional.pad(img, (math.floor(pad_width[1]),math.ceil(pad_width[1]),math.floor(pad_width[0]),math.ceil(pad_width[0])))

class DiscretizeET(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
    
    def __call__(self, masks_labels):
        return torch.nn.AdaptiveMaxPool2d((self.grid_size,self.grid_size))(masks_labels)

class MyKeyboardInterruptionException(Exception):
    "Keyboard Interrupt activate signal handler"
    pass
    
def interupt_handler(signum, frame):
    raise MyKeyboardInterruptionException

signal.signal(signal.SIGINT, interupt_handler)

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

#dataset wrapper to load a dataset to memory for faster batch loading
class LoadToMemory(Dataset):
    def __init__(self, original_dataset, parallel = False):
        super().__init__()
        indices_iterations = np.arange(len(original_dataset))
        if parallel:
            manager = multiprocessing.Manager()
            numpys = manager.list([original_dataset])
            self.list_elements = Parallel(n_jobs=16, batch_size = 1)(delayed(get_one_sample)(list_index,element_index, numpys) for list_index, element_index in enumerate(indices_iterations))
        else:
            self.list_elements = [original_dataset[0]]*len(original_dataset)
            for list_index, element_index in enumerate(indices_iterations): 
                print(f'{element_index}+/{len(original_dataset)}')
                ait(element_index, list_index, original_dataset, self.list_elements)
    def __len__(self):
        return len(self.list_elements)
    
    def __getitem__(self, index):
        return self.list_elements[index]

#dataset wrapper to apply transformations to a pytorch dataset. i defines the index of the element
# of the tuple returned by original_dataset to which the transformation should be applied
class TransformsDataset(Dataset):
    def __init__(self, original_dataset, transform, i=0):
        super().__init__()
        self.original_dataset = original_dataset
        self.transform = transform
        if type(self.transform)==type([]):
            self.transform  = transforms.Compose(self.transform)
        self.i = i
    
    def apply_transform_ith_element(self, batch, transform):        
        to_return = *batch[:self.i], transform(batch[self.i]), *batch[(self.i+1):]
        return to_return
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        return self.apply_transform_ith_element(self.original_dataset[index], self.transform)

class IteratorLoaderDifferentSizesSameBatch:
    def __init__(self, loader1, loader2, n_iter_per_epoch):
        self.loaders = [loader1, loader2]
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

#generic class to save a pytorch dataset to H5 files, and load them to memory if
# they have already been saved. It assumes that filename is a uniue identifier for the dataset content, 
# such that if filename exists, it will directly load the h5 file and not use original_dataset.
# it also saves a pickle file to store the original organization of the dataset, while the h5
# file store the data.
class H5Dataset(Dataset):
    def __init__(self, fn_create_dataset, path = '.', filename = None,individual_datasets = False):
        super().__init__()
        self.individual_datasets = individual_datasets
        original_dataset = None
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
        if filename is None:
            if original_dataset is None:
                original_dataset = fn_create_dataset()
            # if filename is not provided, try to get a hash key for the dataset to characterize its content
            # for several datasets, this will take a really long time, since it has to iterate through the full dataset.
            # it is better to provide your own unique name
            def hash_example(name_, structure, fixed_args):
                structure = np.array(structure)
                structure.flags.writeable = False
                fixed_args['sha1'].update(structure.data)
            sha1 = hashlib.sha1()
            for example in original_dataset:
                apply_function_to_nested_iterators(example, {'sha1': sha1}, hash_example)
            filename = str(sha1.hexdigest())
        filename = filename + self.get_extension()
        self.filepath_h5 = path + '/' + filename
        structure_file = path + '/' + filename + '_structure.pkl'
        length_file = path + '/' + filename + '_length.pkl'
        if not os.path.exists(self.filepath_h5):
            try:
                if original_dataset is None:
                    original_dataset = fn_create_dataset()
                self.len_ = len(original_dataset)
                with self.get_file_handle()(self.filepath_h5, 'w') as h5f:
                    structure = self.create_h5_structure(original_dataset[0], h5f, len(original_dataset), self.individual_datasets)
                    for index in range(len(original_dataset)): 
                        element = original_dataset[index]
                        print(f'{index}/{len(original_dataset)}')
                        self.pack_h5(element, index, h5f, self.individual_datasets)
                with open(structure_file, 'wb') as output:
                    pickle.dump(structure, output, pickle.HIGHEST_PROTOCOL)
            except Exception as err:
                # if there is an error in the middle of writing, delete the generated files
                # to not have corrupted files
                if os.path.exists(self.filepath_h5):
                    if not os.path.isdir(self.filepath_h5):
                        os.remove(self.filepath_h5)
                    else:
                        shutil.rmtree(self.filepath_h5)
                if os.path.exists(structure_file):
                    if not os.path.isdir(structure_file):
                        os.remove(structure_file)
                    else:
                        shutil.rmtree(structure_file)
                raise type(err)('Error while writing hash ' + filename + '. Deleting files ' + self.filepath_h5 + ' and ' + structure_file).with_traceback(err.__traceback__)
        elif not os.path.exists(structure_file):
            if original_dataset is None:
                original_dataset = fn_create_dataset()
            structure = self.create_h5_structure(original_dataset[0], h5f = None, n_images = len(original_dataset), individual_datasets = self.individual_datasets)
            with open(structure_file, 'wb') as output:
                pickle.dump(structure, output, pickle.HIGHEST_PROTOCOL)
        if not os.path.exists(length_file):
            if original_dataset is None:
                original_dataset = fn_create_dataset()
            with open(length_file, 'wb') as output:
                pickle.dump(len(original_dataset), output, pickle.HIGHEST_PROTOCOL)
        self.file = None
        with open(structure_file, 'rb') as input:
            self.structure = pickle.load(input)
        with open(length_file, 'rb') as input:
            self.len_ = pickle.load(input)
    
    def get_file_handle(self):
        return h5py.File
        
    def get_extension(self):
        return '.h5'
    
    def get_individual_with_fn(self, index, fn_get):
        assert(self.individual_datasets)
        if self.file is None:
            self.file = h5py.File(self.filepath_h5, 'r', swmr = True)
        return self.generic_open_individual_h5(self.structure, index, self.file, self.individual_datasets, fn_get)
    
    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.filepath_h5, 'r', swmr = True)
            if not self.individual_datasets:
                assert(len(self.file['root']['_index_0'])==len(self))
        return self.unpack_h5(self.structure, index, self.file, self.individual_datasets)
    
    def __len__(self):
        return self.len_
        
    @staticmethod
    def create_h5_structure(structure, h5f, n_images,individual_datasets):
        def function_(name_, value, fixed_args):
            if fixed_args['h5f'] is not None:
                if type(value) == type('a') or fixed_args['individual_datasets']:
                    pass
                    # return type(value)
                    # fixed_args['h5f'].create_dataset(name_, shape = [fixed_args['n_images']] + [len(value)], dtype = 'S'+str(len(value)))
                elif type(value) == type(np.array([0])):
                    fixed_args['h5f'].create_dataset(name_, shape = [fixed_args['n_images']] + list(np.array(value).shape), dtype = value.dtype)
                else:
                    fixed_args['h5f'].create_dataset(name_, shape = [fixed_args['n_images']] + list(np.array(value).shape))
            return type(value)
        return apply_function_to_nested_iterators(structure, {'n_images':n_images, 'h5f': h5f, 'individual_datasets':individual_datasets}, function_)
    
    @staticmethod
    def pack_h5(structure, index, h5f, individual_datasets):
        def function_(name_, value, fixed_args):
            if type(value) == type('a') or fixed_args['individual_datasets']:
                fixed_args['h5f'].create_dataset(name_+f"@{fixed_args['index']}", data = value)
                return None
                # value = np.array(value).astype('S'+str(len(value)))
            fixed_args['h5f'][name_][fixed_args['index'],...] = value
            return None
        return apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'individual_datasets':individual_datasets}, function_)
    
    @staticmethod
    def generic_open_individual_h5(structure, index, h5f,individual_datasets, fn_get):
        def function_(name_, value, fixed_args):
            return fn_get(fixed_args['h5f'][name_+f"@{fixed_args['index']}"])
        return apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'individual_datasets':individual_datasets},function_)

    @staticmethod
    def shape_h5(structure, index, h5f,individual_datasets):
        def function_(name_, value, fixed_args):
            if value == type('a') or fixed_args['individual_datasets']:
                return fixed_args['h5f'][name_+f"@{fixed_args['index']}"].shape
            return_value = fixed_args['h5f'][name_].shape
            if type(return_value)==np.bytes_:
                return_value = return_value.decode('utf-8')
            return return_value
        return apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'individual_datasets':individual_datasets},function_)

    @staticmethod
    def unpack_h5(structure, index, h5f,individual_datasets):
        def function_(name_, value, fixed_args):
            if value == type('a') or fixed_args['individual_datasets']:
                return fixed_args['h5f'][name_+f"@{fixed_args['index']}"][()]
            return_value = fixed_args['h5f'][name_][fixed_args['index']]
            if type(return_value)==np.bytes_:
                return_value = return_value.decode('utf-8')
            return return_value
        return apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f, 'individual_datasets':individual_datasets},function_)

#auxiliary function to iterate and apply functions to all elements of a variable composed
# of nested variable of these types: list, tuple, dict
# leafs have to be of kind: np.ndarray, int, float, bool, PIL.Image.Image
def apply_function_to_nested_iterators(structure, fixed_args, function_, name_ = "root"):
    if structure is None or isinstance(structure, (np.ndarray, int, bytes, float, bool, PIL.Image.Image, str,type(type(None)))):
        return function_(name_, structure, fixed_args)
    elif isinstance(structure, list) or isinstance(structure, tuple):
        return [apply_function_to_nested_iterators(item, fixed_args, function_, name_ = name_ + "/" + '_index_' + str(index)) for index, item in enumerate(structure)]
    elif isinstance(structure, dict):
        return {key: apply_function_to_nested_iterators(item, fixed_args, function_, name_ = f'{name_}/{key}') for key, item in structure.items()}
    else:
        raise ValueError('Unsuported type: ' + str(type(structure)))

def get_one_sample(list_index,element_index, original_dataset_):
    print(f'{element_index}-/{len(original_dataset_[0])}')
    return original_dataset_[0][element_index]

def ait(element_index, list_index, original_dataset, list_elements):
    list_elements[list_index] = original_dataset[element_index]
    return list_elements

class SeedingPytorchTransformWithID(Dataset):
    def __init__(self, original_dataset, original_transform, seed, indices_to_transform):
        self.seed = seed
        self.previouspythonstate = None
        self.previoustorchstate = None
        self.original_transform = original_transform
        if type(self.original_transform)==type([]):
            self.original_transform  = transforms.Compose(self.original_transform)
        self.indices_to_transform = indices_to_transform
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        x = self.original_dataset[index]
        fn_run_transform = (lambda s,a: s.original_transform(a))
        outsidepythonstate = random.getstate()
        outsidetorchstate = torch.random.get_rng_state()
        to_return = x
        for i in self.indices_to_transform:
            if self.previouspythonstate is None:
                if self.seed is not None:
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
            else:
                random.setstate(self.previouspythonstate)
                torch.random.set_rng_state(self.previoustorchstate)
            to_return = [*to_return[:i], fn_run_transform(self, to_return[i]), *to_return[(i+1):]]
        self.previouspythonstate = random.getstate() 
        self.previoustorchstate = torch.random.get_rng_state()
        random.setstate(outsidepythonstate)
        torch.random.set_rng_state(outsidetorchstate)
        return to_return

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
    if len(index_produce_val_image)>0 and split!='train':
        instantiate_dataset_ = instantiate_dataset
        instantiate_dataset = lambda :ChangeDatasetToIndexList(instantiate_dataset_(), index_produce_val_image)
    return torch.utils.data.DataLoader(dataset=instantiate_dataset(), batch_size=batch_size,
                        shuffle=(split=='train'), num_workers=num_workers, pin_memory=True, drop_last = (split=='train'))
    