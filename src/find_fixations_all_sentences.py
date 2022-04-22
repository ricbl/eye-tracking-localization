# script used to pregenerate heatmaps associated with each sentence from reports from
# the reflacx dataset which were found to contain a mention for the presence of an abnormality,
# according to the modified chexpert-labeler
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import pathlib
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .eyetracking_object import ETDataset
from .eyetracking_dataset import pre_transform_train
import os
from .global_paths import jpg_path, mimic_dir, eyetracking_dataset_path, preprocessed_heatmaps_location,path_chexpert_labels
from .list_labels import translate_et_to_label

create_for_all_sentences = False

# function that rasterizes a Gaussian centered at x,y of axis-aligned standard deviation sx and sy
# into an array of size sizex, sizey
# It only draws the gaussian inside the area defined by shown_rects_image_space
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
    return to_return

# Given a sequence of fixation from a pandas table sequence_table, this function
# draws heatmaps corresponding to all fixations between sentence_start and sentence_end
def create_heatmap(sequence_table, size_x, size_y, sentence_start, sentence_end):
    img = np.zeros((size_y, size_x), dtype=np.float32)
    for index, row in sequence_table.iterrows():
        if not (row['timestamp_end_fixation']>sentence_start and row['timestamp_start_fixation']<=sentence_end):
            continue
        # setting standard deviation of gaussians to one degree of visual field
        angle_circle = 1
        
        shown_rects_image_space = [round(row['xmin_shown_from_image']) ,round(row['ymin_shown_from_image']),round(row['xmax_shown_from_image']),round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'],row['x_position'], row['angular_resolution_y_pixels_per_degree']*angle_circle, row['angular_resolution_x_pixels_per_degree']*angle_circle, size_y,size_x, shown_rects_image_space)
        fixation_start = max(row['timestamp_start_fixation'],sentence_start)
        fixation_end = min(row['timestamp_end_fixation'],sentence_end)
        amplitude_multiplier = fixation_end - fixation_start
        img += gaussian*amplitude_multiplier
    # normalize heatmap to between 0 and 1
    return img/np.max(img)

# this function generates eye-tracking heatmaps for all reports related to a specific chest x-ray.
# The heatmaps are only generated for sentence that contain a positive mention of a label.
# the get_image argument is only True when generating Figure 1 from the paper
def generate_et_heatmaps_for_one_image(phase, trial, image_name,df_this_trial,data_folder,folder_name, get_image = False, method = '5s'):
    index_image = 0
    total_uncut = 0
    total_cut = 0
    total_sentences = 0
    labeled_reports = pd.read_csv(f'{path_chexpert_labels}/labeled_reports_{phase}.csv')
    
    # For every reading of this chest xray
    for index, df_this_index_this_trial in df_this_trial.iterrows():
        
        print('trial', trial, 'index_image', index_image)
        image_size_x = int(float(df_this_index_this_trial['image_size_x']))
        image_size_y = int(float(df_this_index_this_trial['image_size_y']))
        fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
        transcription = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/timestamps_transcription.csv')
        joined_trainscription = ' '.join(transcription['word'].values)
        row_chexpert = labeled_reports.iloc[index]
        if not row_chexpert['Reports'].replace('.',' .').replace(',',' ,') == joined_trainscription:
            print(index)
            print(df_this_index_this_trial["id"])
            print(row_chexpert['Reports'].replace('.',' .').replace(',',' ,'))
            print(joined_trainscription)
        assert(row_chexpert['Reports'].replace('.',' .').replace(',',' ,') == joined_trainscription)
        
        #getting a list of sentences (sentences_indices) that contain at least one label
        sentences_indices = set()
        indices_stop = [index_char for index_char, char in enumerate(row_chexpert['Reports']) if char == '.']
        for et_label in translate_et_to_label:
                if f'{et_label.lower()}_location' in row_chexpert:
                    if len(row_chexpert[f'{et_label.lower()}_location'])>2:
                        for current_range in row_chexpert[f'{et_label.lower()}_location'].strip('][').replace('], [', '],[').split('],['):
                            current_range = current_range.split(',')
                            current_range = [int(item) for item in current_range]
                            for index_sentence, index_stop in enumerate(indices_stop):
                                if current_range[0]<index_stop:
                                    current_sentence = index_sentence
                                    break
                            sentences_indices.add(current_sentence)
        
        if get_image:
            print(joined_trainscription)
            
            # check which words of the previous sentence previous sentence are close than 5 seconds to the start of the current sentence
            print(transcription)
        
        full_sentence = ''
        new_sentence = True
        index_sentence = 0
        len_start = 0
        first_start_timestamp = 0
        timestamp_start_first_sentence = None
        last_end_timestamp = 0
        first_token = True
        for _, row in transcription.iterrows():
            word = row['word']
            
            # adding spaces to the sentence only when it should be added
            if word != '.' and word != ',' and word != ':' and not first_token:
                full_sentence+= ' '
            first_token = False
            if new_sentence:
                first_start_timestamp_previous = first_start_timestamp #start time of the previous sentence
                last_end_timestamp_previous = last_end_timestamp #end time of the previous sentence
                first_start_timestamp = row['timestamp_start_word'] #  start time of the current sentence
                if timestamp_start_first_sentence is None:
                    timestamp_start_first_sentence = first_start_timestamp # start time of the first sentence
                len_start = len(full_sentence)
                new_sentence = False
            full_sentence+= word
            
            #When reaching the end of a sentence, create its heatmap
            if word == '.':
                last_end_timestamp = row['timestamp_end_word'] # end time of current sentence
                
                numpy_filename = f'{folder_name}/{trial}_{index_image}_{index_sentence}.npy'
                
                #only create heatmaps for sentences that contain mention of presence of a label
                if index_sentence in sentences_indices or create_for_all_sentences:
                    
                    #counting of how many heatmaps were calculated using the 5 second limit and how many were calculated using the "beginning of previous sentence" limit
                    if first_start_timestamp_previous>=first_start_timestamp-5:
                        total_uncut +=1
                    else:
                         total_cut +=1
                    total_sentences +=1
                    
                    if not os.path.exists(numpy_filename):
                        if method =='5s': 
                        # default method. For each sentence, the sentence heatmap will contain previous fixations, 
                        # up to the start of the previous sentence or 5 seconds before the start of the current sentence, whatever happens later
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, max(first_start_timestamp-5, first_start_timestamp_previous), last_end_timestamp)
                        if method =='2.5s':
                        # same as the above method, but using 2.5 seconds instead of 5 seconds
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, max(first_start_timestamp-2.5, first_start_timestamp_previous), last_end_timestamp)
                        if method =='7.5s':
                        # same as the above method, but using 7.5 seconds
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, max(first_start_timestamp-7.5, first_start_timestamp_previous), last_end_timestamp)
                        if method == 'single':
                        # the heatmap for the current sentence is defined by only the fixations made during the dictation of the current sentence
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, first_start_timestamp, last_end_timestamp)
                        if method == 'double':
                        # the heatmap for the current sentence is defined by the fixations made during the dictation of the current sentence, the previous sentence, and everything in between
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, first_start_timestamp_previous, last_end_timestamp)
                        if method == 'all':
                        # the heatmap for the current sentence is defined by all the fixations from the start of data aquisition for this report, up to the end of the current sentence
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, 0, last_end_timestamp)
                        if method == 'allsentences':
                        # the heatmap for the current sentence is defined by all the fixations from the start of the first dictated sentence, up to the end of the current sentence
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, timestamp_start_first_sentence, last_end_timestamp)
                        if method == 'andpause':
                        # the heatmap for the current sentence is defined by all the fixations from the end of the dictation of the previous sentence, up to the end of the current sentenc
                            this_img = create_heatmap(fixations,image_size_x, image_size_y, last_end_timestamp_previous, last_end_timestamp)
                        # Saving the array in npy format
                        info_dict = {'np_image': this_img, 'img_path': pre_process_path(image_name), 
                                    'trial': trial, 'id':df_this_index_this_trial["id"],'char_start':len_start, 'char_end':len(full_sentence)-2}
                        if not get_image:
                            np.save(numpy_filename[:-4], info_dict)
                        
                        # imageio.imwrite(f'{folder_name}/{trial}_{index_image}_{index_sentence}.png', this_img) # too slow; only for debug
                else:
                    if os.path.exists(numpy_filename):
                        os.remove(numpy_filename)
                
                new_sentence = True
                index_sentence+=1
        
        with open(f'{data_folder}/{df_this_index_this_trial["id"]}/transcription.txt',mode='r') as transcription_txt:
            full_transcription = transcription_txt.read()
        assert(full_transcription == full_sentence)
        index_image += 1
    if get_image:
        return this_img
    else:
        return total_uncut, total_cut, total_sentences

# creates threads for the parallel generation eye-tracking heatmaps of each chest x-ray of a phase from the REFLACX dataset
def create_heatmaps(phase, data_folder, filename_phase, folder_name='heatmaps', method = '5s'): #method can be '5s','single','double','all','all_sentence'
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    df = pd.read_csv(data_folder + filename_phase)
    df = df[df['eye_tracking_data_discarded']==False]
    df = df.reset_index()
    all_images = df['image'].unique()
    a = Parallel(n_jobs=16)(delayed(generate_et_heatmaps_for_one_image)(phase, trial, image_name,df[df['image']==image_name],data_folder,folder_name, method = method) for trial, image_name in enumerate(sorted(all_images)))
    print(np.array(a).sum(0))

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

# # check all reports from image 0 to 100 to check which one are short enough to fit in Figure 1 and contains non-complex positive labels
def print_100_reports():
    file_phase_3 = 'metadata_phase_3.csv'
    df = pd.read_csv(eyetracking_dataset_path + file_phase_3)
    df = df[df['eye_tracking_data_discarded']==False]
    all_images = df['image'].unique()
    for i in range(0, 100):
        print(i)
        image_name = all_images[i]
        index_image = 0
        df_this_trial = df[df['image']==image_name]
        trial = i
        data_folder = eyetracking_dataset_path
        for _, df_this_index_this_trial in df_this_trial.iterrows():
            print('trial', trial, 'index_image', index_image)
            transcription = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/timestamps_transcription.csv')
            print(' '.join(transcription['word'].values))

# function used to generate the whole heatmap of all readings of a chest x-ray
# only used for the generation of Figure 1 from the paper
def generate_et_heatmaps_for_one_image_full(trial, image_name,df_this_trial,data_folder):
    index_image = 0
    for _, df_this_index_this_trial in df_this_trial.iterrows():
        print('trial', trial, 'index_image', index_image)
        image_size_x = int(float(df_this_index_this_trial['image_size_x']))
        image_size_y = int(float(df_this_index_this_trial['image_size_y']))
        fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
        this_img = create_heatmap_full(fixations,image_size_x, image_size_y)
        index_image += 1
        
    return this_img

# function used to generate the whole heatmap of a reading
# only used for the generation of Figure 1 from the paper
def create_heatmap_full(sequence_table, size_x, size_y):
    img = np.zeros((size_y, size_x), dtype=np.float32)
    for index, row in sequence_table.iterrows():
        angle_circle = 1
        shown_rects_image_space = [round(row['xmin_shown_from_image']) ,round(row['ymin_shown_from_image']),round(row['xmax_shown_from_image']),round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'],row['x_position'], row['angular_resolution_y_pixels_per_degree']*angle_circle, row['angular_resolution_x_pixels_per_degree']*angle_circle, size_y,size_x, shown_rects_image_space)
        img += gaussian*(row['timestamp_end_fixation']-row['timestamp_start_fixation'])
    return img/np.sum(img)

# # generate images for Figure 1 (method) in paper
# draw image with the best report: (index 82 had a small enough positive report)
def get_images_figure_1(index_trial=82, grid_size = 16):
    file_phase_3 = 'metadata_phase_3.csv'
    df = pd.read_csv(eyetracking_dataset_path + file_phase_3)
    df = df[df['eye_tracking_data_discarded']==False]
    all_images = df['image'].unique()
    
    split_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-split.csv')
    mimic_df = pd.read_csv(split_csv)
    image_name = all_images[index_trial]
    
    print('Image file:')
    print(image_name)
    
    # save original image
    map = generate_et_heatmaps_for_one_image(3,index_trial, image_name,df[df['image']==image_name],eyetracking_dataset_path,'delete/', get_image = True)
    plt.imshow(plt.imread(pre_process_path(image_name)), cmap='gray')
    plt.imshow(map, cmap='jet', alpha = 0.3)
    plt.axis('off')
    plt.savefig('./sentence_heatmap_example.png', bbox_inches='tight', pad_inches = 0)
    
    # save original heatmap
    plt.imshow(plt.imread(pre_process_path(image_name)), cmap='gray')
    plt.imshow(generate_et_heatmaps_for_one_image_full(index_trial, image_name,df[df['image']==image_name],eyetracking_dataset_path), cmap='jet', alpha = 0.3)
    plt.axis('off')
    plt.savefig('./full_heatmap_example.png', bbox_inches='tight', pad_inches = 0)
    
    et_dataset = ETDataset( 'train', 3, pre_transform_train)
    
    with open('./val_mimicid.txt', 'r') as txt_val:
        all_val_ids = txt_val.read().splitlines() 
        all_val_ids = [int(item) for item in all_val_ids]
    mimic_df = mimic_df.loc[mimic_df['split'].isin(['train']) & ~mimic_df['subject_id'].isin(all_val_ids)]
    train_df = pd.merge(df,mimic_df).reset_index()
    index_train = train_df[train_df['image']==image_name].index.values[0]
    
    plt.imshow(et_dataset[index_train][0].squeeze(0), cmap='gray')
    plt.axis('off')
    plt.savefig('./input_cxr_example.png', bbox_inches='tight', pad_inches = 0)
    
    print('Image labels:')
    print(et_dataset[index_train][1])
    plt.imshow((torch.nn.AdaptiveMaxPool2d((grid_size,grid_size))(torch.tensor(et_dataset[index_train][2][1]).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).numpy()>0.15)*1., cmap='gray')
    plt.axis('off')
    plt.savefig('./discretized_example.png', bbox_inches='tight', pad_inches = 0)

#generates the heatmaps from all sentences beforehand, to speed up the loading of the dataset later
def pregenerate_all_sentence_heatmaps():
    file_phase_3 = 'metadata_phase_3.csv'
    
    create_heatmaps(3, eyetracking_dataset_path, file_phase_3,
                          folder_name=f'{preprocessed_heatmaps_location}/heatmaps_sentences_phase_3')

if __name__=='__main__':
    # file_phase_1 = 'metadata_phase_1.csv'
    # 
    # create_heatmaps(1, eyetracking_dataset_path, file_phase_1,
    #                   folder_name=f'{preprocessed_heatmaps_location}/heatmaps_sentences_phase_1')
    # 
    # file_phase_2 = 'metadata_phase_2.csv'
    # 
    # create_heatmaps(2,eyetracking_dataset_path, file_phase_2,
    #                     folder_name=f'{preprocessed_heatmaps_location}/heatmaps_sentences_phase_2')
    pregenerate_all_sentence_heatmaps()
    # print_100_reports()
    get_images_figure_1()
