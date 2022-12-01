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
from collections import defaultdict
from copy import deepcopy
import math
from sklearn.linear_model import LogisticRegression
import functools as ft
from pydicom import dcmread

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

def is_inside_ellipse(xpoint, ypoint, ellipse_xmin, ellipse_xmax, ellipse_ymin, ellipse_ymax):
    rad_cc = (((xpoint-(ellipse_xmax+ellipse_xmin)/2)**2)/(((ellipse_xmax-ellipse_xmin)/2.)**2)) + (((ypoint-(ellipse_ymax+ellipse_ymin)/2)**2)/(((ellipse_ymax-ellipse_ymin)/2.)**2))
    return rad_cc <= 1.

def translate_pandas_table(the_table, translation_keys):
    to_return = deepcopy(the_table[set(the_table.columns).difference(translation_keys.keys())])
    to_return.loc[:,list(set( item for key in translation_keys if key in the_table.columns for item in translation_keys[key]))] = False
    for translation_key in translation_keys:
        if translation_key in the_table.columns:
            to_return[translation_keys[translation_key]] = (np.logical_or(to_return[translation_keys[translation_key]], the_table[translation_key].values[:,None]))
    return to_return

def converted_distance_to_screen(point_1, point_2, rectangle_shown_from_image_1, rectangle_in_screen_coordinates_1, rectangle_shown_from_image_2, rectangle_in_screen_coordinates_2):
    point_1_screen_0 = (point_1[0] - rectangle_shown_from_image_1[0])/(rectangle_shown_from_image_1[2] - rectangle_shown_from_image_1[0])*(rectangle_in_screen_coordinates_1[2] - rectangle_in_screen_coordinates_1[0]) + rectangle_in_screen_coordinates_1[0]
    point_2_screen_0 = (point_2[0] - rectangle_shown_from_image_2[0])/(rectangle_shown_from_image_2[2] - rectangle_shown_from_image_2[0])*(rectangle_in_screen_coordinates_2[2] - rectangle_in_screen_coordinates_2[0]) + rectangle_in_screen_coordinates_2[0]
    point_1_screen_1 = (point_1[1] - rectangle_shown_from_image_1[1])/(rectangle_shown_from_image_1[3] - rectangle_shown_from_image_1[1])*(rectangle_in_screen_coordinates_1[3] - rectangle_in_screen_coordinates_1[1]) + rectangle_in_screen_coordinates_1[1]
    point_2_screen_1 = (point_2[1] - rectangle_shown_from_image_2[1])/(rectangle_shown_from_image_2[3] - rectangle_shown_from_image_2[1])*(rectangle_in_screen_coordinates_2[3] - rectangle_in_screen_coordinates_2[1]) + rectangle_in_screen_coordinates_2[1]
    return math.sqrt((point_2_screen_0 - point_1_screen_0)**2 + (point_2_screen_1 - point_1_screen_1)**2)

def apply_windowing(x,level,width):
    return np.minimum(np.maximum(((x.astype(float)-level)/width+0.5),0),1)

def open_dicom(filpath_image_this_trial):
    with dcmread(filpath_image_this_trial) as header:
        max_possible_value = (2**float(header.BitsStored)-1)
        x = header.pixel_array
        x = x.astype(float)/max_possible_value
        if 'WindowWidth' in header:
            if hasattr(header.WindowWidth, '__getitem__'):
                wwidth = header.WindowWidth[0]
            else:
                wwidth = header.WindowWidth
            if hasattr(header.WindowCenter, '__getitem__'):
                wcenter = header.WindowCenter[0]
            else:
                wcenter = header.WindowCenter
            windowing_width = wwidth/max_possible_value
            windowing_level = wcenter/max_possible_value
            if header.PhotometricInterpretation=='MONOCHROME1' or not ('PixelIntensityRelationshipSign' in header) or header.PixelIntensityRelationshipSign==1:
                x = 1-x
                windowing_level = 1 - windowing_level
        else:
             if 'VOILUTSequence' in header:
                lut_center = float(header.VOILUTSequence[0].LUTDescriptor[0])/2
                window_center = find_nearest(np.array(header.VOILUTSequence[0].LUTData), lut_center)
                deltas = []
                for i in range(10,31):
                    deltas.append((float(header.VOILUTSequence[0].LUTData[window_center+i]) - float(header.VOILUTSequence[0].LUTData[window_center-i]))/2/i)
                window_width = lut_center/sum(deltas)*2*len(deltas)
                windowing_width = window_width/max_possible_value
                windowing_level = (window_center-1)/max_possible_value
                if windowing_width < 0:
                    windowing_width = -windowing_width
                    x = 1-x
                    windowing_level = 1 - windowing_level
             else:
                windowing_width = 1
                windowing_level = 0.5;
                if header.PhotometricInterpretation=='MONOCHROME1' or not ('PixelIntensityRelationshipSign' in header) or header.PixelIntensityRelationshipSign==1:
                    x = 1-x
                    windowing_level = 1 - windowing_level
        return apply_windowing(x, windowing_level, windowing_width)

# Given a sequence of fixation from a pandas table sequence_table, this function
# draws heatmaps corresponding to all fixations between sentence_start and sentence_end
def create_heatmap(sequence_table, size_x, size_y, sentence_start, sentence_end, bboxes, dict_label_this_sentence, image_name):
    bboxes_to_use = bboxes[bboxes[dict_label_this_sentence].any(axis = 1)][['xmin','ymin','xmax','ymax', 'certainty']]
    if sentence_start>=sentence_end:
        print(sentence_start,sentence_end)
    assert(sentence_start<sentence_end)
    img = np.zeros((size_y, size_x), dtype=np.float32)
    fixation_properties = []
    for index, row in sequence_table.iterrows():
        
        is_inside = False
        for _,bbox_to_use in bboxes_to_use.iterrows():
            is_inside = is_inside_ellipse(row['x_position'],row['y_position'], bbox_to_use['xmin'], bbox_to_use['xmax'], bbox_to_use['ymin'], bbox_to_use['ymax'])
            if is_inside:
                break
        saccade_pixel_length_1 = None
        saccade_pixel_length_2 = None
        fixation_start = max(row['timestamp_start_fixation'],sentence_start)
        fixation_end = min(row['timestamp_end_fixation'],sentence_end)
        amplitude_multiplier = fixation_end - fixation_start
        
        # normalized_pupil_1 = row['pupil_area_normalized']/sequence_table['pupil_area_normalized'].mean()
        # fixation_properties.append({'property_name':'normalized_pupil_area','property_value':normalized_pupil_1,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
        # 
        # normalized_pupil_2 = math.sqrt(row['pupil_area_normalized'])/np.sqrt(sequence_table['pupil_area_normalized']).mean()
        # fixation_properties.append({'property_name':'normalized_pupil_diameter','property_value':normalized_pupil_2,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
        # mimic_dataset_location = '/usr/sci/scratch/ricbl/mimic-cxr/dicom_dataset/'
        # image = open_dicom(f'{mimic_dataset_location}/{image_name}')
        # 
        # image = image[int(row['ymin_shown_from_image']):int(row['ymax_shown_from_image']),int(row['xmin_shown_from_image']):int(row['xmax_shown_from_image'])]
        # 
        # image = np.minimum(np.maximum((image-row['window_level'])/row['window_width'], 0), 1)
        # brightness = image.mean()*(row['xmax_in_screen_coordinates']-row['xmin_in_screen_coordinates'])*(row['ymax_in_screen_coordinates']-row['ymin_in_screen_coordinates'])/2160/3840
        # normalized_pupil_3 = row['pupil_area_normalized']*brightness
        # fixation_properties.append({'property_name':'pupil_normalized_image','property_value':normalized_pupil_3,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
        
        fixation_properties.append({'property_name':'pupil_area','property_value':row['pupil_area_normalized'],'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
        
        
        fixation_properties.append({'property_name':'fixation_length','property_value':amplitude_multiplier,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
        
        if index>0:
            # saccade_length_1 = (row['timestamp_start_fixation']-sequence_table['timestamp_end_fixation'].iloc[index-1])
            # fixation_properties.append({'property_name':'previous_saccade_time','property_value':saccade_length_1,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
            saccade_pixel_length_1 = converted_distance_to_screen(
                sequence_table[['x_position','y_position']].iloc[index-1].values, 
                row[['x_position','y_position']].values, 
                sequence_table[['xmin_shown_from_image','ymin_shown_from_image','xmax_shown_from_image','ymax_shown_from_image']].iloc[index-1].values, 
                sequence_table[['xmin_in_screen_coordinates','ymin_in_screen_coordinates','xmax_in_screen_coordinates','ymax_in_screen_coordinates']].iloc[index-1].values,
                row[['xmin_shown_from_image','ymin_shown_from_image','xmax_shown_from_image','ymax_shown_from_image']].values, 
                row[['xmin_in_screen_coordinates','ymin_in_screen_coordinates','xmax_in_screen_coordinates','ymax_in_screen_coordinates']].values)/1000
            fixation_properties.append({'property_name':'previous_saccade_pixel','property_value':saccade_pixel_length_1,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
            
        if index<len(sequence_table)-1:
            # saccade_length_2 = (sequence_table['timestamp_start_fixation'].iloc[index+1]-row['timestamp_end_fixation'])
            # fixation_properties.append({'property_name':'next_saccade_time','property_value':saccade_length_2,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
            saccade_pixel_length_2 = converted_distance_to_screen(
                sequence_table[['x_position','y_position']].iloc[index+1].values, 
                row[['x_position','y_position']].values, 
                sequence_table[['xmin_shown_from_image','ymin_shown_from_image','xmax_shown_from_image','ymax_shown_from_image']].iloc[index-1].values, 
                sequence_table[['xmin_in_screen_coordinates','ymin_in_screen_coordinates','xmax_in_screen_coordinates','ymax_in_screen_coordinates']].iloc[index-1].values,
                row[['xmin_shown_from_image','ymin_shown_from_image','xmax_shown_from_image','ymax_shown_from_image']].values, 
                row[['xmin_in_screen_coordinates','ymin_in_screen_coordinates','xmax_in_screen_coordinates','ymax_in_screen_coordinates']].values)/1000
            fixation_properties.append({'property_name':'next_saccade_pixel','property_value':saccade_pixel_length_2,'fixation_length':amplitude_multiplier,'is_inside':is_inside, 'fixation_index': index})
            
        if not (row['timestamp_end_fixation']>sentence_start and row['timestamp_start_fixation']<=sentence_end):
            continue
        # setting standard deviation of gaussians to one degree of visual field
        angle_circle = 1
        
        shown_rects_image_space = [round(row['xmin_shown_from_image']) ,round(row['ymin_shown_from_image']),round(row['xmax_shown_from_image']),round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'],row['x_position'], row['angular_resolution_y_pixels_per_degree']*angle_circle, row['angular_resolution_x_pixels_per_degree']*angle_circle, size_y,size_x, shown_rects_image_space) #CLASSIFIER
        img += gaussian*amplitude_multiplier
        
        # fixation_properties.append({'property_name':,'property_value':,'fixation_length':amplitude_multiplier,'is_inside':is_inside})
    # normalize heatmap to between 0 and 1
    return img/np.max(img),pd.DataFrame(fixation_properties)

# this function generates eye-tracking heatmaps for all reports related to a specific chest x-ray.
# The heatmaps are only generated for sentence that contain a positive mention of a label.
# the get_image argument is only True when generating Figure 1 from the paper
def generate_et_heatmaps_for_one_image(phase, trial, image_name,df_this_trial,data_folder,folder_name, get_image = False, full_method_name = '1.5s_startofsentence_lastmention'):
    fixation_metadata = None
    index_image = 0
    total_uncut = 0
    total_cut = 0
    total_sentences = 0
    labeled_reports = pd.read_csv(f'{path_chexpert_labels}/labeled_reports_{phase}.csv')
    
    time_to_go_back = float(full_method_name.split('s')[0]) if ('.' in full_method_name)  else None
    start_method = full_method_name.split('_')[1] if ('.' in full_method_name) else None
    end_method = full_method_name.split('_')[2] if ('.' in full_method_name) else full_method_name.split('_')[1]
    method = full_method_name.split('_')[0]
    print(full_method_name)
    # For every reading of this chest xray
    for index, df_this_index_this_trial in df_this_trial.iterrows():
        
        print('trial', trial, 'index_image', index_image)
        image_size_x = int(float(df_this_index_this_trial['image_size_x']))
        image_size_y = int(float(df_this_index_this_trial['image_size_y']))
        fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
        transcription = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/timestamps_transcription.csv')
        bboxes = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/anomaly_location_ellipses.csv')
        bboxes = translate_pandas_table(bboxes, translate_et_to_label)
        
        joined_trainscription = ' '.join(transcription['word'].values)
        row_chexpert = labeled_reports.iloc[index]
        if not row_chexpert['Reports'].replace('.',' .').replace(',',' ,') == joined_trainscription:
            print(index)
            print(df_this_index_this_trial["id"])
            print(row_chexpert['Reports'].replace('.',' .').replace(',',' ,'))
            print(joined_trainscription)
        assert(row_chexpert['Reports'].replace('.',' .').replace(',',' ,') == joined_trainscription)
        
        numpy_filename = f'{folder_name}/{trial}_{index_image}.npy'
        if not os.path.exists(numpy_filename):
            this_img, fixation_metadata_this_sentence = create_heatmap(fixations,image_size_x, image_size_y, 0, transcription.iloc[-1]['timestamp_end_word'], bboxes, set(), image_name)
            info_dict = {'np_image': this_img, 'img_path': pre_process_path(image_name), 
                        'trial': trial, 'id':df_this_index_this_trial["id"]}
            if not get_image:
                np.save(numpy_filename[:-4], info_dict)
        #getting a list of sentences (sentences_indices) that contain at least one label
        sentences_indices = set()
        first_range_sentence = {}
        last_range_sentence = {}
        dict_label_each_sentence = defaultdict(set)
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
                            for dest_label in translate_et_to_label[et_label]:
                                dict_label_each_sentence[current_sentence].add(dest_label)
                            if current_sentence in first_range_sentence.keys():
                                if current_range[0]<first_range_sentence[current_sentence]:
                                    first_range_sentence[current_sentence] = current_range[0]
                                if current_range[0]>last_range_sentence[current_sentence]:
                                    last_range_sentence[current_sentence] = current_range[0]
                            else:
                                first_range_sentence[current_sentence] = current_range[0]
                                last_range_sentence[current_sentence] = current_range[0]
        
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
        time_start_mention = None
        time_last_mention = None
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
            
            if index_sentence in sentences_indices:
                assert(first_range_sentence[index_sentence]<=last_range_sentence[index_sentence])
            
            if index_sentence in sentences_indices and len(full_sentence)-len(word)<=first_range_sentence[index_sentence] and len(full_sentence)>=first_range_sentence[index_sentence]:
                time_start_mention = row['timestamp_start_word']
            
            if index_sentence in sentences_indices and len(full_sentence)-len(word)<=last_range_sentence[index_sentence] and len(full_sentence)>=last_range_sentence[index_sentence]:
                time_last_mention = row['timestamp_start_word']
            
            #When reaching the end of a sentence, create its heatmap
            if word == '.':
                last_end_timestamp = row['timestamp_end_word'] # end time of current sentence
                
                numpy_filename = f'{folder_name}/{trial}_{index_image}_{index_sentence}.npy'
                #only create heatmaps for sentences that contain mention of presence of a label
                if index_sentence in sentences_indices or create_for_all_sentences:
                    assert(time_last_mention>=first_start_timestamp)
                    assert(time_last_mention<=last_end_timestamp)
                    assert(time_start_mention<=time_last_mention)
                    
                    #counting of how many heatmaps were calculated using the time limit and how many were calculated using the "beginning of previous sentence" limit
                    if time_to_go_back is not None:
                        if first_start_timestamp_previous>=first_start_timestamp-time_to_go_back:
                            total_uncut +=1
                        else:
                             total_cut +=1
                        total_sentences +=1
                    
                    if end_method=='endofsentence':
                        time_to_end_heatmap = last_end_timestamp
                    elif end_method=='firstmention':
                        time_to_end_heatmap = time_start_mention
                    elif end_method=='lastmention':
                        time_to_end_heatmap = time_last_mention
                    elif end_method=='startofsentence':
                        time_to_end_heatmap = first_start_timestamp
                    
                    if not os.path.exists(numpy_filename):
                        if '.' in method:
                            if start_method=='startofsentence':
                                time_to_count_seconds_from = first_start_timestamp
                            elif start_method=='endofsentence':
                                time_to_count_seconds_from = last_end_timestamp
                            elif start_method=='mention':
                                time_to_count_seconds_from = time_start_mention 
                            # default method. For each sentence, the sentence heatmap will contain previous fixations, 
                            # up to the start of the previous sentence or 1.5 seconds before the start of the current sentence, whatever happens later
                            
                            this_img, fixation_metadata_this_sentence = create_heatmap(fixations,image_size_x, image_size_y, max(time_to_count_seconds_from-time_to_go_back, first_start_timestamp_previous), time_to_end_heatmap, bboxes, dict_label_each_sentence[index_sentence], image_name)
                        if method == 'single':
                        # the heatmap for the current sentence is defined by only the fixations made during the dictation of the current sentence
                            this_img, fixation_metadata_this_sentence = create_heatmap(fixations,image_size_x, image_size_y, first_start_timestamp, time_to_end_heatmap, bboxes, dict_label_each_sentence[index_sentence], image_name)
                        if method == 'double':
                        # the heatmap for the current sentence is defined by the fixations made during the dictation of the current sentence, the previous sentence, and everything in between
                            this_img, fixation_metadata_this_sentence = create_heatmap(fixations,image_size_x, image_size_y, first_start_timestamp_previous, time_to_end_heatmap, bboxes, dict_label_each_sentence[index_sentence], image_name)
                        if method == 'all':
                        # the heatmap for the current sentence is defined by all the fixations from the start of data aquisition for this report, up to the end of the current sentence
                            this_img, fixation_metadata_this_sentence = create_heatmap(fixations,image_size_x, image_size_y, 0, time_to_end_heatmap, bboxes, dict_label_each_sentence[index_sentence], image_name)
                        if method == 'allsentences':
                        # the heatmap for the current sentence is defined by all the fixations from the start of the first dictated sentence, up to the end of the current sentence
                            this_img, fixation_metadata_this_sentence = create_heatmap(fixations,image_size_x, image_size_y, timestamp_start_first_sentence, time_to_end_heatmap, bboxes, dict_label_each_sentence[index_sentence], image_name)
                        if method == 'andpause':
                        # the heatmap for the current sentence is defined by all the fixations from the end of the dictation of the previous sentence, up to the end of the current sentenc
                            this_img, fixation_metadata_this_sentence = create_heatmap(fixations,image_size_x, image_size_y, last_end_timestamp_previous, time_to_end_heatmap, bboxes, dict_label_each_sentence[index_sentence], image_name)
                        
                        fixation_metadata_this_sentence['index_sentence'] = index_sentence
                        fixation_metadata_this_sentence['index_radiologist'] = index
                        if fixation_metadata is not None:
                            fixation_metadata = pd.concat([fixation_metadata, fixation_metadata_this_sentence])
                        else:
                            fixation_metadata = fixation_metadata_this_sentence
                        
                        # Saving the array in npy format
                        info_dict = {'np_image': this_img, 'img_path': pre_process_path(image_name), 
                                    'trial': trial, 'id':df_this_index_this_trial["id"],'char_start':len_start, 'char_end':len(full_sentence)-2}
                        if not get_image:
                            np.save(numpy_filename[:-4], info_dict)
                        
                else:
                    if os.path.exists(numpy_filename):
                        os.remove(numpy_filename)
                
                new_sentence = True
                index_sentence+=1
        
        with open(f'{data_folder}/{df_this_index_this_trial["id"]}/transcription.txt',mode='r') as transcription_txt:
            full_transcription = transcription_txt.read()
        assert(full_transcription == full_sentence)
        index_image += 1
    if fixation_metadata is not None:
        fixation_metadata['trial'] = trial
        fixation_metadata['phase'] = phase
    if get_image:
        return this_img
    else:
        return total_uncut, total_cut, total_sentences, fixation_metadata

# creates threads for the parallel generation eye-tracking heatmaps of each chest x-ray of a phase from the REFLACX dataset
def create_heatmaps(phase, data_folder, filename_phase, folder_name='heatmaps', method = '1.5s_startofsentence_lastmention'): #method can be '5s','single','double','all','all_sentence'
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    df = pd.read_csv(data_folder + filename_phase)
    df = df[df['eye_tracking_data_discarded']==False]
    df = df.reset_index()
    all_images = df['image'].unique()
    a = Parallel(n_jobs=16)(delayed(generate_et_heatmaps_for_one_image)(phase, trial, image_name,df[df['image']==image_name],data_folder,folder_name, full_method_name = method) for trial, image_name in enumerate(sorted(all_images)))
    b = list(map(list, zip(*a)))
    
    if not all(v is None for v in b[3]) :
        table_fixation_properties = pd.concat(b[3])
        table_fixation_properties.to_csv(f'./{phase}_fixation_statistics.csv')
        # all_property_names = ['normalized_pupil_area', 'normalized_pupil_diameter', 'pupil_area', 'fixation_length', 'previous_saccade_time', 'next_saccade_time', 'previous_saccade_pixel', 'next_saccade_pixel']
        all_property_names = ['pupil_area', 'fixation_length', 'previous_saccade_pixel', 'next_saccade_pixel']
        
        columns_to_keep = ['property_value','index_sentence','phase', 'trial', 'index_radiologist', 'is_inside', 'fixation_index']
        columns_to_merge_on = ['index_sentence', 'phase', 'trial', 'index_radiologist', 'is_inside', 'fixation_index']
        list_of_tables_with_each_property = [table_fixation_properties[table_fixation_properties['property_name']==property_name][columns_to_keep].rename(columns={'property_value':property_name}) for property_name in all_property_names]
        transposed_fixation_properties_table = ft.reduce(lambda left, right: pd.merge(left, right, on=columns_to_merge_on, how = 'inner'), list_of_tables_with_each_property)
        X = transposed_fixation_properties_table[all_property_names].values
        y = transposed_fixation_properties_table['is_inside'].values

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
    
    plt.imshow(map, cmap='gray')
    plt.axis('off')
    plt.savefig('./sentence_heatmap_example.png', bbox_inches='tight', pad_inches = 0)
    
    # save original heatmap
    plt.imshow(plt.imread(pre_process_path(image_name)), cmap='gray')
    plt.imshow(generate_et_heatmaps_for_one_image_full(index_trial, image_name,df[df['image']==image_name],eyetracking_dataset_path), cmap='jet', alpha = 0.3)
    plt.axis('off')
    plt.savefig('./full_heatmap_example_w_xray.png', bbox_inches='tight', pad_inches = 0)
    
    
    
    et_dataset = ETDataset( 'train', 3, pre_transform_train, False)
    
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
    file_phase_1 = 'metadata_phase_1.csv'
    
    create_heatmaps(1, eyetracking_dataset_path, file_phase_1,
                      folder_name=f'{preprocessed_heatmaps_location}/heatmaps_sentences_phase_1')
    
    file_phase_2 = 'metadata_phase_2.csv'
    
    create_heatmaps(2,eyetracking_dataset_path, file_phase_2,
                        folder_name=f'{preprocessed_heatmaps_location}/heatmaps_sentences_phase_2')
    pregenerate_all_sentence_heatmaps()
    # print_100_reports()
    get_images_figure_1()
