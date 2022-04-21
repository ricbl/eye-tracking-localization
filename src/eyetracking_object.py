# file containing the ETDataset class, which defines how chest x-rays, eye-tracking heatmaps, and ground truth ellipses are loaded together
import pandas as pd
import numpy as np
import torch
import os
import imageio
from torchvision import transforms
from skimage.draw import ellipse
from .list_labels import list_labels, translate_mimic_to_label, str_labels_mimic, translate_et_to_label
from .global_paths import jpg_path, path_chexpert_labels, mimic_dir, metadata_et_location, preprocessed_heatmaps_location

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

# given an ellipse represented by its coordinates, draws an image of im_size 
# (xsize, ysize) containing that ellipse.
# Ellipse coordinates should be given as the extreme points of the axes of the ellipse 
# (rect = [xmin,ymin,xmax,ymax])
def get_ellipses_image(rect, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    rr, cc = ellipse((rect[0]+rect[2])/2, (rect[1]+rect[3])/2, abs(rect[0]-rect[2])/2, abs(rect[1]-rect[3])/2,im_size)
    img_mask[rr, cc] = 1
    return np.transpose(img_mask)

class ETDataset(torch.utils.data.Dataset):
    def __init__(self, split, phase, pretransform, preprocessed_heatmaps_location_ = preprocessed_heatmaps_location):
        # mimic tables
        label_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-chexpert.csv')
        split_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-split.csv')
        mimic_df = pd.read_csv(split_csv)
        self.pretransform = transforms.Compose(pretransform)
        
        #load subject_ids that should be moved from training set to validation set
        with open('./val_mimicid.txt', 'r') as txt_val:
            all_val_ids = txt_val.read().splitlines() 
            all_val_ids = [int(item) for item in all_val_ids]
        
        if split == 'train':
            #remove loaded subject_ids from training
            mimic_df = mimic_df.loc[mimic_df['split'].isin(['train']) & ~mimic_df['subject_id'].isin(all_val_ids)]
        elif split == 'val':
            #add loaded subject_ids to validation
            mimic_df = mimic_df.loc[mimic_df['split'].isin(['validate']) | mimic_df['subject_id'].isin(all_val_ids)]
        elif split == 'test':
            mimic_df = mimic_df[mimic_df['split'] == 'test']
        mimic_df = pd.merge(mimic_df, pd.read_csv(label_csv))
        
        #convert labels to bool (-1 and +1 are considered label present)
        mimic_df[str_labels_mimic] = (mimic_df[str_labels_mimic].astype('float').abs().values > 0) * 1.
        self.mimic_labels = mimic_df
        
        # reflacx tables
        self.labels_table_reflacx = pd.read_csv(f'{path_chexpert_labels}/labeled_reports_{phase}.csv')
        self.metadata_table_reflacx = pd.read_csv(f'{metadata_et_location}/metadata_phase_{phase}.csv')
        
        #remove cases that do not have eye tracking data
        self.metadata_table_reflacx = self.metadata_table_reflacx[self.metadata_table_reflacx['eye_tracking_data_discarded']==False]
        
        #get a sorted  list of all the mimic-cxr images included in the reflacx dataset
        all_images = self.metadata_table_reflacx['image'].unique()
        self.list_images_et_maps = sorted(all_images)
        
        # apply transfer from train to validation of the same ids as above
        assert(len(self.labels_table_reflacx) == len(self.metadata_table_reflacx))
        self.metadata_table_reflacx = self.metadata_table_reflacx.reset_index()
        self.labels_table_reflacx = self.labels_table_reflacx.reset_index()
        if split=='test':
            indices_keep = self.metadata_table_reflacx['split'].isin(['test'])
            self.labels_table_reflacx = self.labels_table_reflacx.loc[indices_keep]
            self.metadata_table_reflacx = self.metadata_table_reflacx.loc[indices_keep]
        else:
            with open('./val_mimicid.txt', 'r') as txt_val:
                all_val_ids = txt_val.read().splitlines() 
                all_val_ids = [int(item) for item in all_val_ids]
            if split=='val':
                indices_keep = self.metadata_table_reflacx['subject_id'].isin(all_val_ids)
            else:
                assert(split=='train')
                indices_keep = ~self.metadata_table_reflacx['subject_id'].isin(all_val_ids) & ~self.metadata_table_reflacx['split'].isin(['test'])
            self.metadata_table_reflacx = self.metadata_table_reflacx.loc[indices_keep]
            self.labels_table_reflacx = self.labels_table_reflacx.loc[indices_keep]
        assert(len(self.labels_table_reflacx) == len(self.metadata_table_reflacx))
        self.metadata_et_location = metadata_et_location
        
        # path where the heatmaps, preprocessed by the find_fixations_all_sentences file, are
        self.et_maps_path = f'{preprocessed_heatmaps_location_}/heatmaps_sentences_phase_{phase}/'
        
        self.phase = phase
    
    def __len__(self):
        return len(self.labels_table_reflacx)
    
    def __getitem__(self,index):
        # get the table rows associated with this specific chest x-ray
        row_chexpert = self.labels_table_reflacx.iloc[index]
        row_metadata = self.metadata_table_reflacx.iloc[index]
        
        #load chest x-ray and its report
        img = imageio.imread(pre_process_path(row_metadata['image']))
        assert(len(img.shape)==2)
        img = self.pretransform(img)
        assert(len(img.shape)==3)
        with open(f'{self.metadata_et_location}/{row_metadata["id"]}/transcription.txt', 'r') as txt_file:
            report_original = txt_file.readlines()[0]
        assert(report_original==row_chexpert['Reports'])
        
        # rasterize ellipses associated with each label to ellipse_labels
        # and the image-level label given by radiologists to ellipse_labels
        # (if at least an ellipse is present for a label, ellipse_labels is true for that label)
        df_box = pd.read_csv(f'{self.metadata_et_location}/{row_metadata["id"]}/anomaly_location_ellipses.csv')
        imsize = [None, None]
        imsize[0] = row_metadata['image_size_x']
        imsize[1] = row_metadata['image_size_y']
        ellipse_labels = np.zeros([len(list_labels),img.shape[1], img.shape[2]])
        image_level_gt = np.zeros([len(list_labels)])
        # for each box
        for _, box in df_box.iterrows():
            # only consider label present if certainty is Possibly or higher
            if box['certainty']>2:
                # calculate rasterized ellipse
                ellipse_map = get_ellipses_image(box[['xmin','ymin','xmax','ymax']].values, imsize)
                resized_ellipse_map = (self.pretransform(ellipse_map) >0.5)*1.
                for et_label in translate_et_to_label:
                    if et_label in box.keys():
                        # check if the ellipse is associated with et_label
                        if box[et_label]:
                            for destination_label in translate_et_to_label[et_label]:
                                image_level_gt[list_labels.index(destination_label)] = 1.
                                # since a label may have more than one ellipse associated
                                # with it, mix the multiple 
                                # occurrences using the maximum function
                                ellipse_labels[list_labels.index(destination_label)] = np.maximum(resized_ellipse_map.squeeze(0), ellipse_labels[list_labels.index(destination_label)])
        
        # calculates the heatmaps from eye-tracking (masks_labels) and
        # labels extracted from reports (image_level_labels)
        indices_stop = [index_char for index_char, char in enumerate(report_original) if char == '.']
        masks_labels = np.zeros([len(list_labels),img.shape[1], img.shape[2]])
        image_level_labels = np.zeros([len(list_labels)])
        loaded_sentences = {}
        for et_label in translate_et_to_label:
            #iterate through all labels present in the table of labels
            # extracted from reports
            if f'{et_label.lower()}_location' in row_chexpert:
                # Checking if table cell is longer than 2 because for labels 
                # that were not found in the report, the table will contain 
                # range "[]", so if it is larger than two, the label was found
                # in the report
                if len(row_chexpert[f'{et_label.lower()}_location'])>2:
                    # iterate through each mention of a label in a specific report
                    for current_range in row_chexpert[f'{et_label.lower()}_location'].strip('][').replace('], [', '],[').split('],['):
                        current_range = current_range.split(',')
                        # current_range is a list with two int numbers representing 
                        # the starting index and ending index of a mention of 
                        # the label in the report string
                        current_range = [int(item) for item in current_range]
                        
                        # iterate through the location of all sentences to find out in which sentence this mention was mentioned
                        for index_sentence, index_stop in enumerate(indices_stop):
                            if current_range[0]<index_stop:
                                current_sentence = index_sentence
                                break
                        assert(current_range[1]<=indices_stop[current_sentence])
                        if current_sentence>0:
                            assert(current_range[0]>indices_stop[current_sentence-1])
                        
                        # Only run the code of loading a sentence heatmap if that sentence heatmap was not still loaded for this chest x-ray
                        if not current_sentence in loaded_sentences:
                            index_et_map = self.list_images_et_maps.index(row_metadata['image'])
                            if self.phase<3:
                                subindex_et_map = self.metadata_table_reflacx[self.metadata_table_reflacx['image']==row_metadata['image']]['id'].values.tolist().index(row_metadata['id'])
                            else:
                                subindex_et_map = 0
                            loaded_pickle_base = np.load(f'{self.et_maps_path}/{index_et_map}_{subindex_et_map}_{current_sentence}.npy', allow_pickle=True).item()
                            assert(loaded_pickle_base['img_path']==pre_process_path(row_metadata['image']))
                            assert(loaded_pickle_base['id']==row_metadata['id'])
                            assert(loaded_pickle_base['char_start']<=current_range[0])
                            assert(loaded_pickle_base['char_end']+1>=current_range[1])
                            et_map = loaded_pickle_base['np_image']
                            resized_et_map = self.pretransform(et_map)
                            loaded_sentences[current_sentence] = resized_et_map.squeeze(0)
                        
                        # associate loaded sentence heatmap with all destination_label labels associated with et_label
                        for destination_label in translate_et_to_label[et_label]:
                            # since a mention of a label may happen more than 
                            # once in the same report, mix the multiple 
                            # occurrences using the maximum function
                            masks_labels[list_labels.index(destination_label)] = np.maximum(loaded_sentences[current_sentence],masks_labels[list_labels.index(destination_label)])
                            image_level_labels[list_labels.index(destination_label)] = 1
        
        # get the ground truth labels provided by the mimic-cxr dataset
        mimic_gt_row = self.mimic_labels[self.mimic_labels['dicom_id']==row_metadata["dicom_id"]]
        mimic_gt = np.zeros([len(list_labels)])
        for et_label in translate_mimic_to_label:
            # if any of the et_labels translated to destination_label are present, destination_label is present
            for destination_label in translate_mimic_to_label[et_label]:
                if mimic_gt_row[et_label].values[0]>0:
                    mimic_gt[list_labels.index(destination_label)]= 1.
        
        # dataset outputs by index:
        # 0: chest x-ray; size (1,512,512)
        # 1: image level labels, as extracted from the report by the modified 
        # chexpert-labeler; size (10,)
        # 2: per-label eye-tracking heatmaps, representing the fixations during 
        # the sentence of that label; size (10,512,512)
        # 3: per-label spatial binary representation of the ellipses drawn by
        # radiologists to localize abnormalities; size (10,512,512)
        # 4: image level labels, as selected by radiologists; positive when 
        # there is an ellipse preset for that lable, negative when there isn't;
        # size (10,)
        # 5: image level labels according to the MIMIC dataset; size (10,)
        return np.array(img), \
                image_level_labels,\
                np.array(masks_labels),\
                np.array(ellipse_labels),\
                image_level_gt,\
                mimic_gt