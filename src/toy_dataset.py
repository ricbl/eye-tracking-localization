import numpy as np
import random
import math
import torchvision
import cv2
import imageio
from scipy.stats import multivariate_normal
import time
from .find_fixations_all_sentences import get_gaussian
from torch.utils.data import Dataset
import pathlib
from .eyetracking_dataset import get_dataset_et
from .mimic_dataset import get_dataset
from skimage.draw import ellipse

start_time = time.time()

position_coordinates = {
'top': [100,100,412,150],
'bottom': [100,362,412,412],
'middle': [100,150,412,412],
'center':[200,250,412,412]
}

size_in_pixels = {'small':30, 'medium':60, 'big':120}
position = ['top', 'bottom', 'middle', 'middle', 'middle', 'middle', 'center', 'center']
size = ['small', 'small', 'small', 'medium', 'medium', 'small', 'big', 'medium']
order_of_appearance_ = [0, 0, 1, 1, 1, 3, 2, 2]
chance_of_presence = [0.03,0.3,0.05,0.215,0.215,0.02,0.2,0.04]
chance_of_negative = [0.6, 0, 0, 0, 0, 0.4, 0.6, 0.6]
chance_of_two_boxes = [0,0.5,0.5,0,0.25,0.25,0,0,0]
confounder = [0.5,0,0,0,0,0,0,0]
list_labels = [2,3,4,5,6,7,8,9]
list_opacities = [1,4,5,6]
distraction_label = 0
distraction_label_probability = 0.25
distraction_label_samples = 6
noise_level = 0.2
n_samples_annotated = 1000
n_samples_unannotated = 10000

joined_labels = {6:[0.5,1], 5:[0.5,1]}

def get_ellipses_image(rect, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    rr, cc = ellipse((rect[0]+rect[2])/2, (rect[1]+rect[3])/2, abs(rect[0]-rect[2])/2, abs(rect[1]-rect[3])/2,im_size)
    img_mask[rr, cc] = 1
    return np.transpose(img_mask)

def get_gaussian(y,x,sy,sx, sizey,sizex, shown_rects_image_space):
    mu = [y-shown_rects_image_space[1],x-shown_rects_image_space[0]]
    sig = [sy**2,sx**2]
    x = np.arange(0, shown_rects_image_space[2]-shown_rects_image_space[0], 1)
    y = np.arange(0, shown_rects_image_space[3]-shown_rects_image_space[1], 1)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 1] = X
    pos[:, :, 0] = Y
    to_return = np.zeros([sizey,sizex])
    to_return[shown_rects_image_space[1]:shown_rects_image_space[3], shown_rects_image_space[0]:shown_rects_image_space[2]] = multivariate_normal(mu, sig).pdf(pos)
    return to_return/to_return.max()

#negative labels: all, random, none
class ToyDataset(Dataset):
    def __init__(self, split, unannotated, negative_labels = 'random', order_of_appearance = order_of_appearance_):
        # self.all_image_level_labels = np.zeros([10000, 10])
        # self.localizations = np.zeros([10000, 10, 512,512])
        self.unannotated = unannotated
        self.split = split
        self.image_path = f'./toy_dataset/' if not unannotated else f'./toy_dataset_unannotated/'
        # return
        pathlib.Path(f'{self.image_path}/{split}/').mkdir(parents=True, exist_ok=True) 
        
        outsidepythonstate = random.getstate()
        # outsidetorchstate = torch.random.get_rng_state()
        outsidenumpystate = np.random.get_state()
        
        if not unannotated:
            if split == 'train':
                np.random.seed(9854)
                random.seed(7663)
            elif split=='val':
                np.random.seed(465465)
                random.seed(148488)
            elif split == 'test':
                np.random.seed(98798)
                random.seed(33543688)
        else:
            if split == 'train':
                np.random.seed(6536847)
                random.seed(499972)
            elif split=='val':
                np.random.seed(654324669)
                random.seed(9849876535)
            elif split == 'test':
                np.random.seed(479849887)
                random.seed(323216468)
        
        self.all_heatmaps = []
        
        self. all_image_level_labels = []
        self.localizations = []
        
        mnist_dataset = torchvision.datasets.MNIST(root = './', train = not (split == 'test'), download = True)
        indices_mnist = list(range(len(mnist_dataset)))
        if split!='test':
            valid_size = 0.1
            num_train = len(indices_mnist)
            index_split = int(valid_size * num_train)
            # get a fixed random validation set for every run
            np.random.shuffle(indices_mnist)
            indices_mnist = {'train':indices_mnist[index_split:], 'val':indices_mnist[:index_split]}[split]
        
        for index_sample in range(n_samples_annotated if not self.unannotated or split!='train' else n_samples_unannotated):
            print(index_sample)
            # sample presence for all labels
            label_present = np.random.binomial(1,chance_of_presence, [len(chance_of_presence)])

            #sample how many distractions
            n_distractions = np.random.binomial(distraction_label_samples,distraction_label_probability)
            
            # get indices that appear and shuffle them
            ordered_labels = np.array(list_labels)[np.random.permutation(np.where(label_present)).tolist()[0]].tolist()
            
            if negative_labels=='all':
                ordered_negatives = np.array(list_labels).tolist()
            elif negative_labels=='random':
                ordered_negatives = np.array(list_labels)[np.random.permutation(np.where(np.random.binomial(1,chance_of_negative, [len(chance_of_negative)]))).tolist()[0]].tolist()
            elif negative_labels=='none':
                ordered_negatives = []
            
            ordered_negatives = [item for item in ordered_negatives if item not in ordered_labels]
            
            #initialize variable to check if canvas has been drawn to in some region
            drawn_to = np.zeros([512,512])
            
            current_image = np.zeros([512,512])
            localization = np.zeros([10,512,512])
            current_heatmaps = [[] for i in range(max(order_of_appearance)+1)]
            current_labels = [[] for i in range(max(order_of_appearance)+1)]
            
            # draw indices in shuffled order, using the position coordinates and size
            for current_label in ordered_labels + [distraction_label] + ordered_negatives:
                if current_label==distraction_label:
                    n_drawings = n_distractions
                    side_in_pixels  = size_in_pixels[size[random.randint(0,len(size)-1)]]
                    coordinates = [side_in_pixels,side_in_pixels,512-side_in_pixels,512-side_in_pixels]
                    label_to_draw = distraction_label
                else:
                    label_to_draw = current_label
                    # if current labels is one of the labels that can be joined, sample if it should be joined
                    if current_label in joined_labels:
                        if random.random()<joined_labels[current_label][0]:
                            label_to_draw = joined_labels[current_label][1]
                    
                    # check if label should appear twice
                    if random.random()<chance_of_two_boxes[list_labels.index(current_label)]:
                        n_drawings = 2
                    else:
                        n_drawings = 1
                    
                    side_in_pixels  = size_in_pixels[size[list_labels.index(current_label)]]
                    coordinates = position_coordinates[position[list_labels.index(current_label)]]
                if current_label in ordered_labels and random.random()<confounder[list_labels.index(current_label)]:
                    coordinates_cofounder = [25,25,75,75]
                    assert(drawn_to[coordinates_cofounder[0]:coordinates_cofounder[2],coordinates_cofounder[1]:coordinates_cofounder[3]].sum()==0)
                    current_image[coordinates_cofounder[0]:coordinates_cofounder[2],coordinates_cofounder[1]:coordinates_cofounder[3]] = 255
                    drawn_to[coordinates_cofounder[0]:coordinates_cofounder[2],coordinates_cofounder[1]:coordinates_cofounder[3]] = 1
                if current_label!=distraction_label:
                    one_hot_labels = np.zeros(10).astype(np.float32)
                    if current_label in ordered_labels:
                        one_hot_labels[current_label] = 1
                        if current_label in list_opacities:
                            one_hot_labels[1] = 1
                        if current_label!=label_to_draw:
                            for key in joined_labels:
                                if joined_labels[key][1]==label_to_draw:
                                    one_hot_labels[key] = 1
                    current_labels[order_of_appearance[list_labels.index(current_label)]].append(one_hot_labels)
                    current_heatmap = np.zeros([512,512])
                for index_drawing in range(n_drawings):
                    n_tries = 0
                    # sample up to 10 times
                    while n_tries<10:
                        coordinate_0 = random.randint(coordinates[0],coordinates[2])
                        coordinate_1 = random.randint(coordinates[1],coordinates[3])
                        current_coordinates = [math.ceil(coordinate_0-side_in_pixels/2), math.ceil(coordinate_1-side_in_pixels/2), math.ceil(coordinate_0+side_in_pixels/2), math.ceil(coordinate_1+side_in_pixels/2)]
                        
                        # check if region sampled was already drawn to
                        if current_label in ordered_labels or current_label==distraction_label: 
                            if drawn_to[current_coordinates[0]:current_coordinates[2],current_coordinates[1]:current_coordinates[3]].sum()>0:
                                n_tries += 1
                                continue
                        
                            # if not, draw number
                            # get next sample of that class
                            n_tries_get_digit = 0
                            while n_tries_get_digit< 10000:
                                sampled_index = random.choice(indices_mnist)
                                n_tries_get_digit += 1
                                if mnist_dataset[sampled_index][1] == label_to_draw:
                                    break
                            mnist_digit = np.array(mnist_dataset[sampled_index][0])
                            resized_digit = cv2.resize(mnist_digit, dsize=(current_coordinates[2]-current_coordinates[0], current_coordinates[3]-current_coordinates[1]), interpolation=cv2.INTER_LINEAR)
                            current_image[current_coordinates[0]:current_coordinates[2],current_coordinates[1]:current_coordinates[3]] = resized_digit
                            drawn_to[current_coordinates[0]:current_coordinates[2],current_coordinates[1]:current_coordinates[3]] = 1
                        if current_label!=distraction_label and not self.unannotated:
                            gaussian = get_gaussian(coordinate_0,coordinate_1,side_in_pixels,side_in_pixels, 512,512, [0,0,512,512])
                            current_heatmap += gaussian/gaussian.max()
                            ellipses = get_ellipses_image([round(coordinate_0-side_in_pixels/2),round(coordinate_1-side_in_pixels/2),round(coordinate_0+side_in_pixels/2),round(coordinate_1+side_in_pixels/2)], [512,512])
                            if current_label!=label_to_draw:
                                for key in joined_labels:
                                    if joined_labels[key][1]==label_to_draw:
                                        localization[key] = np.maximum(ellipses, localization[key])
                            else:
                                localization[current_label] = np.maximum(ellipses, localization[current_label])
                        break
                if current_label!=distraction_label and not self.unannotated:
                    
                    current_heatmaps[order_of_appearance[list_labels.index(current_label)]].append(current_heatmap)
            #add some noise
            current_image = current_image/2+255/4 + cv2.randn(current_image/2+255/4,(0),(30)) 
            if not self.unannotated:
                flat_current_heatmaps = [item for sublist in current_heatmaps for item in sublist]
                if len(flat_current_heatmaps)==0:
                    flat_current_heatmaps += [np.zeros([512,512])]
                self.all_heatmaps.append(np.stack(flat_current_heatmaps))
            flat_current_labels = [item for sublist in current_labels for item in sublist]
            if len(flat_current_labels)==0:
                flat_current_labels += [np.zeros(10).astype(np.float32)]
            flat_current_labels = np.stack(flat_current_labels)
            if self.unannotated:
                flat_current_labels = flat_current_labels.max(0)
                
            self.all_image_level_labels.append(flat_current_labels)
            self.localizations.append(localization)
            imageio.imwrite(f'{self.image_path}/{self.split}/im_{index_sample}.jpg', (current_image).astype('uint8'))
            # imageio.imwrite(f'{self.image_path}/{self.split}/hm_{index_sample}.jpg', (self.all_heatmaps[-1].sum(0)*255/(self.all_heatmaps[-1].sum(0).max()+1e-20)).astype('uint8'))
        random.setstate(outsidepythonstate)
        # torch.random.set_rng_state(outsidetorchstate)
        np.random.set_state(outsidenumpystate)
    
    def __len__(self):
        return len(self.all_image_level_labels)
    
    def __getitem__(self, index):
        cxr = imageio.imread(f'{self.image_path}/{self.split}/im_{index}.jpg')/255.
        if not self.unannotated:
            sequence_image_level_labels = self.all_image_level_labels[index]
            sequence_annotations = self.all_heatmaps[index]
            return cxr*255, sequence_image_level_labels, sequence_annotations, self.localizations[index], np.max(sequence_image_level_labels, axis = 0), np.max(sequence_image_level_labels, axis = 0)
        else:
            sequence_image_level_labels = self.all_image_level_labels[index]
            return cxr*255, sequence_image_level_labels

def get_toy_dataset(split, data_aug_seed, grid_size=8, use_et = True, use_data_aug = False, crop = False, load_to_memory = False, dataset_type = 'h5', sequence_model = False, annotated = True, type_toy = 'original'):
    h5_filename = 'toy_dataset'
    if not annotated:
        h5_filename += '_unannotated'
    if type_toy=='original':
        order_of_appearance = order_of_appearance_
        negative_labels = 'random'
    elif type_toy=='inverted':
        order_of_appearance = order_of_appearance_
        max_order = max(order_of_appearance)
        for i in range(len(order_of_appearance)):
            order_of_appearance[i] = max_order-order_of_appearance[i]
        negative_labels = 'random'
        h5_filename += '_inverted'
    elif type_toy=='no_negatives':
        order_of_appearance = order_of_appearance_
        negative_labels = 'none'
        h5_filename += '_no_negatives'
    elif type_toy=='fixed_sentences':
        order_of_appearance = [0, 1, 2, 3, 4, 5, 6, 7]
        negative_labels = 'all'
        h5_filename += '_fixed_sentences'
    if annotated:
        return get_dataset_et(split, data_aug_seed, grid_size, use_et, use_data_aug, crop, load_to_memory, 'h5', sequence_model = True, h5_filename = h5_filename, fn_initialize_function = lambda split, sequence_model: ToyDataset(split, not annotated, negative_labels, order_of_appearance))
    else:
        return get_dataset(split, data_aug_seed, use_data_aug, crop, h5_filename = h5_filename, fn_create_dataset = lambda split: ToyDataset(split, not annotated, negative_labels, order_of_appearance))

if __name__=='__main__':
    a = get_dataset_et('train', None, 16, True, True, False, False, 'h5', True, h5_filename = 'toy_dataset', fn_initialize_function = lambda split, sequence_model: ToyDataset(split, False)) #ToyDataset('train', False)
    for i in range(len(a[0])):
        print(a[0][i].shape)