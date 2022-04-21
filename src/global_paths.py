import os

h5_path = './' #Location where hdf5 files with preprocessed datasets will be saved

jpg_path = './mimic_images/'  # Location of the files folder of the MIMIC-CXR dataset
mimic_dir = './mimic_tables/' # Location of the tables mimic-cxr-2.0.0-chexpert.csv and mimic-cxr-2.0.0-split.csv from the MIMIC-CXR-JPG dataset
path_chexpert_labels = './' #location of the files containing the labels for the reports of the REFLACX dataset when using the modified chexpert-labeler
metadata_et_location = './main_data/' # location of the metadata tables of the REFLACX dataset
eyetracking_dataset_path = './main_data/' # location of the main content of the REFLACX dataset
preprocessed_heatmaps_location = './' # location where the folders containing precalculated heatmaps are saved