import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import csv
import pathlib
from joblib import Parallel, delayed
import shutil
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .eyetracking_object import ETDataset
from .eyetracking_dataset import pre_transform_train
import os
from .global_paths import jpg_path, mimic_dir, eyetracking_dataset_path
from .list_labels import translate_et_to_label

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

def create_heatmap(sequence_table, size_x, size_y, sentence_start, sentence_end):
    img = np.zeros((size_y, size_x), dtype=np.float32)
    for index, row in sequence_table.iterrows():
        if (row['timestamp_end_fixation']>sentence_start and row['timestamp_start_fixation']<=sentence_end):
            pass
        else:
            continue
        angle_circle = 1
        shown_rects_image_space = [round(row['xmin_shown_from_image']) ,round(row['ymin_shown_from_image']),round(row['xmax_shown_from_image']),round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'],row['x_position'], row['angular_resolution_y_pixels_per_degree']*angle_circle, row['angular_resolution_x_pixels_per_degree']*angle_circle, size_y,size_x, shown_rects_image_space)
        img += gaussian*(min(row['timestamp_end_fixation'],sentence_end)-max(row['timestamp_start_fixation'],sentence_start))
    return img/np.max(img)

def generate_et_heatmaps_for_one_image_full(trial, image_name,df_this_trial,data_folder,folder_name):
    index_image = 0
    for _, df_this_index_this_trial in df_this_trial.iterrows():
        
        print('trial', trial, 'index_image', index_image)
        image_size_x = int(float(df_this_index_this_trial['image_size_x']))
        image_size_y = int(float(df_this_index_this_trial['image_size_y']))
        fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
        this_img = create_heatmap_full(fixations,image_size_x, image_size_y)

        # Log the progress
        with open('logs.csv', 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([trial, folder_name])
        index_image += 1
        # break
    return this_img

def create_heatmap_full(sequence_table, size_x, size_y):
    img = np.zeros((size_y, size_x), dtype=np.float32)
    for index, row in sequence_table.iterrows():
        angle_circle = 1
        shown_rects_image_space = [round(row['xmin_shown_from_image']) ,round(row['ymin_shown_from_image']),round(row['xmax_shown_from_image']),round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'],row['x_position'], row['angular_resolution_y_pixels_per_degree']*angle_circle, row['angular_resolution_x_pixels_per_degree']*angle_circle, size_y,size_x, shown_rects_image_space)
        img += gaussian*(row['timestamp_end_fixation']-row['timestamp_start_fixation'])
    return img/np.sum(img)

def generate_et_heatmaps_for_one_image(trial, image_name,df_this_trial,data_folder,folder_name, get_image = False):
    index_image = 0
    total_uncut = 0
    total_cut = 0
    total_sentences = 0
    labeled_reports = pd.read_csv(f'labeled_reports_3.csv')
    for index, df_this_index_this_trial in df_this_trial.iterrows():
        
        print('trial', trial, 'index_image', index_image)
        image_size_x = int(float(df_this_index_this_trial['image_size_x']))
        image_size_y = int(float(df_this_index_this_trial['image_size_y']))
        fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
        transcription = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/timestamps_transcription.csv')
        joined_trainscription = ' '.join(transcription['word'].values)
        row_chexpert = labeled_reports.iloc[index]
        if not row_chexpert['Reports'].replace('.',' .').replace(',',' ,') == joined_trainscription:
            print(row_chexpert['Reports'].replace('.',' .').replace(',',' ,'))
            print(joined_trainscription)
        assert(row_chexpert['Reports'].replace('.',' .').replace(',',' ,') == joined_trainscription)
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
        last_end_timestamp = 0
        first_token = True
        sentences_dict = []
        for _, row in transcription.iterrows():
            word = row['word']
            if word != '.' and word != ',' and word != ':' and not first_token:
                full_sentence+= ' '
                if not new_sentence:
                    this_sentence+= ' '
            first_token = False
            if new_sentence:
                first_start_timestamp_previous = first_start_timestamp
                last_end_timestamp_previous = last_end_timestamp
                first_start_timestamp = row['timestamp_start_word']
                len_start = len(full_sentence)
                new_sentence = False
                this_sentence = ''
            full_sentence+= word
            this_sentence += word
            if word == '.':
                if index_sentence in sentences_indices:
                    if first_start_timestamp_previous>=first_start_timestamp-5:
                        total_uncut +=1
                    else:
                         total_cut +=1
                    total_sentences +=1
                    this_img = create_heatmap(fixations,image_size_x, image_size_y, max(first_start_timestamp-5, first_start_timestamp_previous), last_end_timestamp)
                    # this_img = create_heatmap(fixations,image_size_x, image_size_y, first_start_timestamp_previous, last_end_timestamp)
                    # this_img = create_heatmap(fixations,image_size_x, image_size_y, first_start_timestamp, last_end_timestamp)
                    
                    # Saving the array in npy format
                    info_dict = {'np_image': this_img, 'img_path': pre_process_path(image_name), 
                                'trial': trial, 'id':df_this_index_this_trial["id"],'char_start':len_start, 'char_end':len(full_sentence)-2}
                    if not get_image:
                        np.save(f'{folder_name}/{trial}_{index_image}_{index_sentence}', info_dict)
                    
                    # imageio.imwrite(f'{folder_name}/{trial}_{index_image}_{index_sentence}.png', this_img) # too slow; only for debug
                
                new_sentence = True
                sentences_dict.append({'index':index_sentence, 'sentence':this_sentence, 'timestamps':[first_start_timestamp, last_end_timestamp]})
                index_sentence+=1
            
            last_end_timestamp = row['timestamp_end_word']
        
        # Open a file: file
        with open(f'{data_folder}/{df_this_index_this_trial["id"]}/transcription.txt',mode='r') as transcription_txt:
            full_transcription = transcription_txt.read()
        if not get_image:
            pass
            # with open(f'{folder_name}/{trial}_{index_image}.txt', 'w') as convert_file:
            #     convert_file.write(json.dumps(sentences_dict, sort_keys=True, indent=4))
            # shutil.copy(pre_process_path(image_name), f'{folder_name}/{trial}_{index_image}.png')
        assert(full_transcription == full_sentence)
        index_image += 1
    if get_image:
        return this_img
    else:
        return total_uncut, total_cut, total_sentences

def create_heatmaps(data_folder, filename_phase, folder_name='heatmaps'):
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    df = pd.read_csv(data_folder + filename_phase)
    df = df[df['eye_tracking_data_discarded']==False]
    all_images = df['image'].unique()
    # generate_et_heatmaps_for_one_image(0, sorted(all_images)[0],df[df['image']==sorted(all_images)[0]],data_folder,folder_name)
    a = Parallel(n_jobs=32)(delayed(generate_et_heatmaps_for_one_image)(trial, image_name,df[df['image']==image_name],data_folder,folder_name) for trial, image_name in enumerate(sorted(all_images)))
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

# # generate images for Figure 1 (method) in paper
# draw image with the best report:
def get_images_figure_1(index_trial=82):
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
    map = generate_et_heatmaps_for_one_image(index_trial, image_name,df[df['image']==image_name],eyetracking_dataset_path,'delete/', get_image = True)
    plt.imshow(plt.imread(pre_process_path(image_name)), cmap='gray')
    plt.imshow(map, cmap='jet', alpha = 0.3)
    plt.axis('off')
    plt.savefig('./sentence_heatmap_example.png', bbox_inches='tight', pad_inches = 0)
    
    # save original heatmap
    plt.imshow(plt.imread(pre_process_path(image_name)), cmap='gray')
    plt.imshow(generate_et_heatmaps_for_one_image_full(index_trial, image_name,df[df['image']==image_name],eyetracking_dataset_path,'delete/'), cmap='jet', alpha = 0.3)
    plt.axis('off')
    plt.savefig('./full_heatmap_example.png', bbox_inches='tight', pad_inches = 0)
    
    a = ETDataset( 'train', 3, pre_transform_train)
    
    with open('./val_mimicid.txt', 'r') as txt_val:
        all_val_ids = txt_val.read().splitlines() 
        all_val_ids = [int(item) for item in all_val_ids]
    mimic_df = mimic_df.loc[mimic_df['split'].isin(['train']) & ~mimic_df['subject_id'].isin(all_val_ids)]
    train_df = pd.merge(df,mimic_df).reset_index()
    index_train = train_df[train_df['image']==image_name].index.values[0]
    
    plt.imshow(a[index_train][0].squeeze(0), cmap='gray')
    plt.axis('off')
    plt.savefig('./input_cxr_example.png', bbox_inches='tight', pad_inches = 0)
    
    print('Image labels:')
    print(a[index_train][1])
    plt.imshow((torch.nn.AdaptiveMaxPool2d((16,16))(torch.tensor(a[index_train][2][1]).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).numpy()>0.15)*1., cmap='gray')
    plt.axis('off')
    plt.savefig('./discretized_example.png', bbox_inches='tight', pad_inches = 0)

#generates the heatmaps from all sentences beforehand, to speed up the loading of the dataset later
def pregenerate_all_sentence_heatmaps():
    file_phase_3 = 'metadata_phase_3.csv'
    
    create_heatmaps(eyetracking_dataset_path, file_phase_3,
                          folder_name='heatmaps_sentences_phase_3')

if __name__=='__main__':
    pregenerate_all_sentence_heatmaps()
    # print_100_reports()
    get_images_figure_1()
