from h5_dataset.h5_dataset import H5Dataset, H5ProcessingFunction, H5ComposeFunction, change_np_type_fn, PackBitArray, UnpackBitArray
import numpy as np

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
        fixed_args['self'].create_individual_dataset_with_data(fixed_args['h5f'], f"{name_}_{self.label_name}_indices_present@{fixed_args['index']}", indices_present)
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
            lambda: fixed_args['self'].get_individual_from_name(fixed_args['h5f'], f"{name_}_{self.label_name}_indices_present@{fixed_args['index']}"))
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
        fixed_args['self'].create_individual_dataset_with_data(fixed_args['h5f'], f"{name_}_{self.label_name}_saved_max_@{fixed_args['index']}", max_value)
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
            lambda: fixed_args['self'].get_individual_from_name(fixed_args['h5f'], f"{name_}_{self.label_name}_saved_max_@{fixed_args['index']}"))
        return max_value * assignment_value

#this dataset returns a list of arrays for each case:
# index 0: chest x-ray; size (1,512,512)
# index 1: image level labels, as extracted from the report by the modified 
# chexpert-labeler; size (10,)
# index 2: per-label eye-tracking heatmaps, representing the fixations during 
# the sentence of that label; size (10,512,512)
# index 3: per-label spatial binary representation of the ellipses drawn by
# radiologists to localize abnormalities; size (10,512,512)
# index 4: image level labels, as selected by radiologists; positive when 
# there is an ellipse preset for that lable, negative when there isn't;
# size (10,)
# index 5: image level labels according to the MIMIC dataset; size (10,)
my_dataset = ETDataset('train')

preprocessing_functions = 
    [change_np_type_fn(np.ubyte, 1), # change the precision of the chest x-ray to 8 bit since it was load from a jpg image and only has 8 bits of precision anyway
    None,
    H5ComposeFunction( # do a sequence of pretransforms
        [save_max_fn('mask_labels'), # Since the maximum of the annotations are not always 1, save the maximum and renormalize it to range [0,1] for maximum use of the numerical precision
         delete_zeros_fn('mask_labels'), # delete the channels that are all zeros in this annotation
         change_np_type_fn(np.ushort, 65535.)] # Store this label with a 16 bit precision since it is not clear how much precision is needed
    ),
    H5ComposeFunction( # do a sequence of pretransforms
        [delete_zeros_fn('ellipse_labels'), # first delete the channels that are all zeros in this annotation
        change_np_type_fn(np.bool, 1.), # change the type of the image to bool, since it is a binary label for each pixel
        PackBitArray('ellipse_labels')]), #Change the storing of the boolean into 1 bit per pixel instead of 8 bits per pixel
    None, 
    None]

# postprocessing functions that undo what the preprocessing functions did
postprocessing_functions = 
    [change_np_type_fn(np.float32, 1./255.), 
    None, 
    H5ComposeFunction(
        [change_np_type_fn(np.float32, 1./65535.), 
        load_max_fn('mask_labels'), 
        reinsert_zeros_fn('mask_labels')]), 
    H5ComposeFunction(
        [UnpackBitArray('ellipse_labels'), 
        change_np_type_fn(np.float32, 1.),
        reinsert_zeros_fn('ellipse_labels')]),
    None,
    None]

#load to memory all parts of the dataset, since it is relatively small (2.1 GB)
load_to_memory = [False, True, True, True, True, True]

# Since the annotation have some channels deleted during preprocessing, they do
# not have the same dimensions for each case. They then have to be stores as
# individual arrays instead of one single array for the whole dataset.
# individual_datasets is then True for the indices of the annotations
individual_datasets = [False, False, True, True, False, False]

trainset = H5Dataset(path = '/scratch/ricbl/', 
                    filename = 'my_dataset',
                    fn_create_dataset = lambda: my_dataset, 
                    individual_datasets = individual_datasets,
                    preprocessing_functions = preprocessing_functions,
                    postprocessing_functions = postprocessing_functions,  
                    n_processes = 8, # use 8 threads for loading my_dataset and preprocessing it
                    load_to_memory = load_to_memory)


    
