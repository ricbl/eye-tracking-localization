import pandas as pd
import os
from .global_path import jpg_path, mimic_dir

label_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-chexpert.csv')
label_df = pd.read_csv(label_csv)

split_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-split.csv')
split_df = pd.read_csv(split_csv)

# ### Merge split df and label_df
# To get a single df with label(s), split, image_name for each image
images_df = pd.merge(left=label_df, right=split_df, left_on = ['subject_id', 'study_id'], right_on=['subject_id', 'study_id'], how='inner')

# ## Filtering out only the good images
# Good images are the ones that are providied in the list image_all_paths.txt
def getImgList(image_path, jpg_path):
    image_list_jpg = []
    with open(image_path) as f:
        image_list = f.readlines()
    for path in image_list:
        temp_path = jpg_path + '/physionet.org/files/' + path.split('files')[-1]
        temp_path = temp_path.replace('.dcm', '.jpg')
        image_list_jpg.append(temp_path.strip())
    return image_list_jpg

image_all_paths = 'image_all_paths.txt'
image_all_paths_jpg = getImgList(image_all_paths, jpg_path)

# ### Get a filtered df with only good paths
good_dicom_ids = []

for i in range(len(image_all_paths_jpg)):
    dc_id = image_all_paths_jpg[i].split('.jpg')[0].split('/')[-1]
    good_dicom_ids.append(dc_id)
images_df_filtered = images_df[images_df['dicom_id'].isin(good_dicom_ids)]
# ### Attaching the path for each good image
path_df = pd.DataFrame({'path': image_all_paths_jpg, 'dicom_id' : good_dicom_ids})
final_df = pd.merge(left=images_df_filtered, right=path_df, on='dicom_id')

# ### Rearanging columns
cols = final_df.columns.to_list()
cols = cols[:2] + cols[-3:] + cols[2:-3]
final_df = final_df[cols]
final_df.head()

# ### Split final_df into Train, Validation and Test dfs
with open('./val_mimicid.txt', 'r') as txt_val:
    all_val_ids = txt_val.read().splitlines() 
    all_val_ids = [int(item) for item in all_val_ids]
train_df = final_df.loc[final_df['split'].isin(['train']) & ~final_df['subject_id'].isin(all_val_ids)]
train_df = train_df.drop('split', axis=1)

val_df = final_df.loc[final_df['split'].isin(['validate']) | final_df['subject_id'].isin(all_val_ids)]
val_df = val_df.drop('split', axis=1)

test_df = final_df[final_df['split'] == 'test']
test_df = test_df.drop('split', axis=1)

train_df.to_csv('train_df.csv', index=False)
val_df.to_csv('val_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

