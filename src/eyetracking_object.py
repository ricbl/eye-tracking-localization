import pandas as pd
import numpy as np
import torch
import os
import imageio
from torchvision import transforms
from skimage.draw import ellipse
from .list_labels import list_labels, translate_mimic_to_label, str_labels_mimic, translate_et_to_label
from .global_paths import jpg_path, path_chexpert_labels, mimic_dir, metadata_et_location

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

def get_ellipses_image(rect, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    rr, cc = ellipse((rect[0]+rect[2])/2, (rect[1]+rect[3])/2, abs(rect[0]-rect[2])/2, abs(rect[1]-rect[3])/2,im_size)
    img_mask[rr, cc] = 1
    return np.transpose(img_mask)

class ETDataset(torch.utils.data.Dataset):
    def __init__(self, split, phase, pretransform):
        label_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-chexpert.csv')
        split_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-split.csv')
        mimic_df = pd.read_csv(split_csv)
        self.pretransform = transforms.Compose(pretransform)
        with open('./val_mimicid.txt', 'r') as txt_val:
            all_val_ids = txt_val.read().splitlines() 
            all_val_ids = [int(item) for item in all_val_ids]
        if split == 'train':
            mimic_df = mimic_df.loc[mimic_df['split'].isin(['train']) & ~mimic_df['subject_id'].isin(all_val_ids)]
        elif split == 'val':
            mimic_df = mimic_df.loc[mimic_df['split'].isin(['validate']) | mimic_df['subject_id'].isin(all_val_ids)]
        elif split == 'test':
            mimic_df = mimic_df[mimic_df['split'] == 'test']
        mimic_df = pd.merge(mimic_df, pd.read_csv(label_csv))
        mimic_df[str_labels_mimic] = (mimic_df[str_labels_mimic].astype('float').abs().values > 0) * 1.
        
        self.mimic_labels = mimic_df
        #open chexpert table
        
        self.labels_table = pd.read_csv(f'{path_chexpert_labels}/labeled_reports_{phase}.csv')
        self.metadata_table = pd.read_csv(f'{metadata_et_location}/metadata_phase_{phase}.csv')
        self.metadata_table = self.metadata_table[self.metadata_table['eye_tracking_data_discarded']==False]
        all_images = self.metadata_table['image'].unique()
        self.list_images_et_maps = sorted(all_images)
        assert(len(self.labels_table) == len(self.metadata_table))
        self.metadata_table = self.metadata_table.reset_index()
        self.labels_table = self.labels_table.reset_index()
        if split=='test':
            indices_keep = self.metadata_table['split'].isin(['test'])
            self.labels_table = self.labels_table.loc[indices_keep]
            self.metadata_table = self.metadata_table.loc[indices_keep]
        else:
            with open('./val_mimicid.txt', 'r') as txt_val:
                all_val_ids = txt_val.read().splitlines() 
                all_val_ids = [int(item) for item in all_val_ids]
            if split=='val':

                indices_keep = self.metadata_table['subject_id'].isin(all_val_ids)
            else:
                assert(split=='train')
                indices_keep = ~self.metadata_table['subject_id'].isin(all_val_ids) & ~self.metadata_table['split'].isin(['test'])
            self.metadata_table = self.metadata_table.loc[indices_keep]
            self.labels_table = self.labels_table.loc[indices_keep]
        assert(len(self.labels_table) == len(self.metadata_table))
        self.metadata_et_location = metadata_et_location
        self.et_maps_path = f'./heatmaps_sentences_phase_{phase}/'
        self.phase = phase
    
    def __len__(self):
        return len(self.labels_table)
    
    def __getitem__(self,index):
        row_chexpert = self.labels_table.iloc[index]
        row_metadata = self.metadata_table.iloc[index]
        
        img = imageio.imread(pre_process_path(row_metadata['image']))
        chest_box = pd.read_csv(f'{self.metadata_et_location}/{row_metadata["id"]}/chest_bounding_box.csv').values[0]
        tmp = chest_box[0]
        chest_box[0] = chest_box[1]
        chest_box[1] = tmp
        tmp = chest_box[2]
        chest_box[2] = chest_box[3]
        chest_box[3] = tmp
        assert(len(img.shape)==2)
        img = self.pretransform(img)
        with open(f'{self.metadata_et_location}/{row_metadata["id"]}/transcription.txt', 'r') as txt_file:
            report_original = txt_file.readlines()[0]
        assert(report_original==row_chexpert['Reports'])
        
        df_box = pd.read_csv(f'{self.metadata_et_location}/{row_metadata["id"]}/anomaly_location_ellipses.csv')
        imsize = [None, None]
        imsize[0] = row_metadata['image_size_x']
        imsize[1] = row_metadata['image_size_y']
        ellipse_labels = np.zeros([len(list_labels),img.shape[1], img.shape[2]])
        image_level_gt = np.zeros([len(list_labels)])
        # for each box
        for _, box in df_box.iterrows():
            if box['certainty']>2:
                # calculate area of the ellipsis
                ellipse_map = get_ellipses_image(box[['xmin','ymin','xmax','ymax']].values, imsize)
                resized_ellipse_map = self.pretransform(ellipse_map)
                for et_label in translate_et_to_label:
                    if et_label in box.keys():
                        if box[et_label]:
                            for destination_label in translate_et_to_label[et_label]:
                                image_level_gt[list_labels.index(destination_label)] = 1.
                                ellipse_labels[list_labels.index(destination_label)] = np.maximum(resized_ellipse_map.squeeze(0), ellipse_labels[list_labels.index(destination_label)])
        
        indices_stop = [index_char for index_char, char in enumerate(report_original) if char == '.']
        masks_labels = np.zeros([len(list_labels),img.shape[1], img.shape[2]])
        image_level_labels = np.zeros([len(list_labels)])
        loaded_sentences = {}
        
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
                        assert(current_range[1]<=indices_stop[current_sentence])
                        if current_sentence>0:
                            assert(current_range[0]>indices_stop[current_sentence-1])
                        if not current_sentence in loaded_sentences:
                            index_et_map = self.list_images_et_maps.index(row_metadata['image'])
                            if self.phase<3:
                                subindex_et_map = self.metadata_table[self.metadata_table['image']==row_metadata['image']]['id'].values.tolist().index(row_metadata['id'])
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
                        for destination_label in translate_et_to_label[et_label]:
                            masks_labels[list_labels.index(destination_label)] = np.maximum(loaded_sentences[current_sentence],masks_labels[list_labels.index(destination_label)])
                            image_level_labels[list_labels.index(destination_label)] = 1
        
        mimic_gt_row = self.mimic_labels[self.mimic_labels['dicom_id']==row_metadata["dicom_id"]]
        mimic_gt = np.zeros([len(list_labels)])
        for et_label in translate_mimic_to_label:
            for destination_label in translate_mimic_to_label[et_label]:
                if mimic_gt_row[et_label].values[0]>0:
                    mimic_gt[list_labels.index(destination_label)]= 1.
        return np.array(img), image_level_labels,np.array(masks_labels), np.array(ellipse_labels),image_level_gt,mimic_gt