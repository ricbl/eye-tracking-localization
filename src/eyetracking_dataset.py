# file defining all data processing applied to the eye-tracking dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from .utils_dataset import TransformsDataset, SeedingPytorchTransformSeveralElements, GrayToThree, ResizeHeatmap, ToTensorMine, XRayResizerPadRound32, ToNumpy, ToTensor1, get_count_positive_labels, get_average_value
from h5_dataset.h5_dataset import H5Dataset, H5ProcessingFunction, H5ComposeFunction, change_np_type_fn, PNGDataset, MMapDataset, ZarrDataset, PackBitArray, UnpackBitArray
from .eyetracking_object import ETDataset
import pandas as pd
import numpy as np
from .global_paths import h5_path
from .list_labels import list_labels
import imageio
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# defining how the chest x-rays are resized (with padding)
pre_transform_train = [ToTensor1(), XRayResizerPadRound32(512), transforms.Resize(512, antialias=True), ToNumpy()]

# defining data augmentations
post_transform_train = [
                        transforms.RandomAffine(degrees=45, translate=(0.15, 0.15),
                        scale=(0.85, 1.15), fill=0),
                        ]

class AddEndToken(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        element = self.original_dataset[index]
        #pad to 12 symbols
        element[1] = torch.nn.functional.pad(element[1], [0,2])
        
        #change the last item of the last item in the sequence to represent an end token
        element[1][-1, -1] = 1.
        return element

# preprocessing function used to delete all channels of annotations of the dataset that are 
# just zeros. It also saves which channels were kept such that the original
# annotation can be recovered when loading the data.
class delete_zeros_fn(H5ProcessingFunction):
    def __init__(self, label_name):
        super().__init__()
        self.label_name = label_name
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        import numpy as np
        indices_present = (np.sum(assignment_value.reshape([assignment_value.shape[0], -1]), axis = 1)>0)*1.
        fixed_args['self'].create_individual_dataset_with_data(fixed_args['h5f'], f"{name_}_{self.label_name}_indices_present/@{fixed_args['index']}", indices_present)
        return np.delete(assignment_value, np.where(1-indices_present), axis = 0)

# postprocessing function used to recover the all-zero channels that were 
# removed by delete_zeros_fn, using the saved indices of kept channels to know 
# which channels were removed
class reinsert_zeros_fn(H5ProcessingFunction):
    def __init__(self, label_name):
        super().__init__()
        self.label_name = label_name
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        import numpy as np
        indices_present = fixed_args['self'].load_variable_to_memory(joined_structures['load_to_memory'], f"{name_}_{self.label_name}_indices_present", fixed_args['index'],
            lambda: np.zeros([len(fixed_args['self']), 10]),
            lambda: fixed_args['self'].get_individual_from_name(fixed_args['h5f'], f"{name_}_{self.label_name}_indices_present/@{fixed_args['index']}"))
        if assignment_value.shape[0]==0:
            return np.zeros([10,assignment_value.shape[1],assignment_value.shape[2]]).astype(assignment_value.dtype)
        indices_where_to_insert = np.where(1-indices_present)[0]
        # using range-offsetted indices for insert (https://stackoverflow.com/questions/52346241/better-way-to-insert-elements-into-numpy-array)
        return np.insert(assignment_value,indices_where_to_insert-np.arange(len(indices_where_to_insert)), 0., axis = 0)

# this function assumes that the minimum value of assignment_value is around 0.
# It is used to save the maximum value of an array, and normalize the range of the
# array to be between 0 and 1, such that the change_np_type_fn can be easily used 
# with a multiplier to cover the whole range of the data type
class save_max_fn(H5ProcessingFunction):
    def __init__(self, label_name):
        super().__init__()
        self.label_name = label_name
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        import numpy as np
        max_value = np.max(assignment_value)
        if max_value==0:
            max_value = 1.
        fixed_args['self'].create_individual_dataset_with_data(fixed_args['h5f'], f"{name_}_{self.label_name}_saved_max_/@{fixed_args['index']}", max_value)
        return assignment_value/max_value

# this function assumes that assignment_value is an array between 0 and 1 and 
# it recovers its original range by loading the saved maximum value
class load_max_fn(H5ProcessingFunction):
    def __init__(self, label_name):
        super().__init__()
        self.label_name = label_name
    def __call__( self, name_, assignment_value, fixed_args, joined_structures):
        import numpy as np
        max_value = fixed_args['self'].load_variable_to_memory(joined_structures['load_to_memory'], f"{name_}_{self.label_name}_saved_max", fixed_args['index'],
            lambda: np.zeros([len(fixed_args['self'])]), 
            lambda: fixed_args['self'].get_individual_from_name(fixed_args['h5f'], f"{name_}_{self.label_name}_saved_max_/@{fixed_args['index']}"))
        return max_value * assignment_value

# this function packs all the arguments needed to load the eye-tracking dataset from the H5 file
# to avoid copy-and-paste of code
def get_h5_dataset(filename, individual_datasets, class_eyetracking_dataset, load_to_memory, fn_create_dataset, sequence_model):
    
    # applying a few optimizations when saving the hdf5 file:
    # - the chest x-ray is transformed to ubyte type since it only contains 255 levels of grey, anyway.
    # This resizing reduces the size of the arrays to 1/4 of what they would be if using float32. No 
    # range change is necessary because chest x-rays are already in the [0,255] range
    # - the eye-tracking heatmaps go through have 3 steps for optimization
        # - the max value of each heatmap is saved, and the range of the heatmap is converted to be
        # from 0 to 1. This allows the heatmap to occupy the full range of the ushort type
        # - all heatmaps that are completely 0 are deleted to reduce the size of the saved arrays. The indices
        # of the heatmaps that were kept are saved to the h5 file so that the original array can be restored
        # - the range is cahnge to use the full range of the ushort type
    # - the ellipse ground truths only have two level of intensities, so beside removing he maps that are all 0, 
    # the type is converted to bool and the array is packed into bit. The packing into bits is necessary for
    # the use of only a single bit per pixel, since arrays of type bool are not efficient and use 8 bits per pixel
    if not sequence_model:
        preprocessing_functions_heatmap = [save_max_fn('mask_labels'), delete_zeros_fn('mask_labels'), change_np_type_fn(np.ushort, 65535.)]
    else:
        preprocessing_functions_heatmap = [save_max_fn('mask_labels'), change_np_type_fn(np.ushort, 65535.)]
    
    if not sequence_model:
        postprocessing_functions_heatmap = [change_np_type_fn(np.float32, 1./65535.), load_max_fn('mask_labels'), reinsert_zeros_fn('mask_labels')]
    else:
        postprocessing_functions_heatmap = [change_np_type_fn(np.float32, 1./65535.), load_max_fn('mask_labels')]
    
    return TransformsDataset(TransformsDataset(
        class_eyetracking_dataset(path = h5_path, 
            filename = filename,
            fn_create_dataset = fn_create_dataset, 
            individual_datasets = individual_datasets,
            preprocessing_functions = [change_np_type_fn(np.ubyte, 1), None, 
                H5ComposeFunction(preprocessing_functions_heatmap), 
                H5ComposeFunction([delete_zeros_fn('ellipse_labels'), change_np_type_fn(np.bool, 1.), PackBitArray('ellipse_labels')])
                , None, None],
            postprocessing_functions = [change_np_type_fn(np.float32, 1./255.), None, 
                H5ComposeFunction(postprocessing_functions_heatmap), 
                H5ComposeFunction([UnpackBitArray('ellipse_labels'), change_np_type_fn(np.float32, 1.), reinsert_zeros_fn('ellipse_labels')])
                , None, None],  n_processes = 16, load_to_memory = [load_to_memory, True, True, True, True, True]),
        [GrayToThree(), ToTensorMine()], indices_to_transform = 0),
        [ToTensorMine()], indices_to_transform = [1,2,3,4,5])

def get_dataset_et(split, data_aug_seed, grid_size=8, use_et = True, use_data_aug = False, crop = False, load_to_memory = False, dataset_type = 'h5', sequence_model = False, h5_filename = 'joint_dataset_et_3_noseg_optimized', calculate_label_specific_heatmaps = True, fn_initialize_function = (lambda split, sequence_model, calculate_label_specific_heatmaps: ETDataset(split,3, pre_transform_train, sequence_model, calculate_label_specific_heatmaps = calculate_label_specific_heatmaps))):
    class_eyetracking_dataset = {'h5':H5Dataset, 'mmap':MMapDataset, 'zarr':ZarrDataset, 'png':PNGDataset}[dataset_type]
    if dataset_type in {'h5','mmap','zarr'}:
        #setting individual_datasets to True for the eye-tracking heatmaps and ellipse ground truth
        # because the optimization with delete_zeros_fn makes these arrays, for each chest x-ray,
        # to have different sizes
        individual_datasets = [False, sequence_model, True, True, False, False]
    elif dataset_type == 'png':
        # the class PNGDataset needs inidividual datasets
        individual_datasets = True
    if split == 'test':
        imageset = TransformsDataset(TransformsDataset(get_h5_dataset('test_' + h5_filename + ('_sequence' if sequence_model else '') + ('_full_heatmap' if not calculate_label_specific_heatmaps else ''),individual_datasets, class_eyetracking_dataset, load_to_memory, lambda: fn_initialize_function('test', sequence_model, calculate_label_specific_heatmaps), sequence_model),
            [normalize], indices_to_transform= 0), [ResizeHeatmap(grid_size)], indices_to_transform = 2)
    if split == 'val':
        imageset = TransformsDataset(TransformsDataset(get_h5_dataset('val_' + h5_filename + ('_sequence' if sequence_model else '') + ('_full_heatmap' if not calculate_label_specific_heatmaps else ''),individual_datasets, class_eyetracking_dataset, load_to_memory, lambda: fn_initialize_function('val', sequence_model, calculate_label_specific_heatmaps), sequence_model),
            [normalize], indices_to_transform = 0), [ResizeHeatmap(grid_size)], indices_to_transform = 2)
    if split == 'train':
        imageset = get_h5_dataset('train_' + h5_filename + ('_sequence' if sequence_model else '') + ('_full_heatmap' if not calculate_label_specific_heatmaps else ''),individual_datasets, class_eyetracking_dataset, load_to_memory, lambda: fn_initialize_function('train', sequence_model, calculate_label_specific_heatmaps), sequence_model)
        if sequence_model:
            imageset = AddEndToken(imageset)
        if not use_data_aug:
            imageset = TransformsDataset(TransformsDataset(imageset, [normalize], indices_to_transform = 0), [ResizeHeatmap(grid_size)], indices_to_transform = 2)
            if not use_et:
                # if training the ellipse model, the ellipses should be resized for training
                imageset = TransformsDataset(imageset, [ResizeHeatmap(grid_size)], indices_to_transform = 3)
        else:
            # if data augmentation is present, it should be performed before the resizing of the heatmaps.
            # SeedingPytorchTransformSeveralElements allows for augmentation random seeds to be the same for all elements in index 0, 2 and 3
            imageset = TransformsDataset(TransformsDataset(SeedingPytorchTransformSeveralElements(imageset,
            post_transform_train, data_aug_seed, indices_to_transform = [0,2,3]  ), [normalize], indices_to_transform = 0), [ResizeHeatmap(grid_size)], indices_to_transform = 2)
            if not use_et:
                # if training the ellipse model, the ellipses should be resized for training
                imageset = TransformsDataset(imageset, [ResizeHeatmap(grid_size)], indices_to_transform = 3)
    
    return imageset

# get list of all grid values in eye-tracking heatmaps that are larger than 0.005
# and save them to 'test_histogram_grid.csv' so that a histogram of the values can be ploted
# and analyzed for deciding the threshold used for the binarizing the eye-tracking heatmaps
def get_list_of_values_from_grid_for_histogram(grid_size = 16):
    c = []
    for phase in [3]:
        a = get_h5_dataset('val_joint_dataset_et_3_noseg_optimized',[False, False, True, True, False, False], H5Dataset, False, False, False)
        for i in range(len(a)):
            d = a[i]
            e = ResizeHeatmap(grid_size)(torch.tensor(d[2])).numpy().reshape([-1])
            for j in range(len(e)):
                if e[j]>0.005:
                    b = {}
                    b['grid_cell'] = e[j]
                    c.append(b)
            print(i)
    pd.DataFrame(c).to_csv('test_histogram_grid.csv')

if __name__=='__main__':
    # code used to print the size and number of positive labels in each split of the reflacx dataset
    grid_size=16
    dataset = get_dataset_et('train', 1, grid_size=grid_size, use_et = True, use_data_aug = True, crop = False, sequence_model = False)
    
    print('Train')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_average_value(dataset)
    dataset = get_dataset_et('val', 1, grid_size=grid_size, use_et = True, use_data_aug = True, crop = False, sequence_model = False)
    
    print('Val')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_average_value(dataset)
    dataset = get_dataset_et('test', 1, grid_size=grid_size, use_et = True, use_data_aug = True, crop = False, sequence_model = False)
    print('Test')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    get_average_value(dataset)
    
