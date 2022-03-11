import imageio
import numpy as np
from torch.utils.data import Dataset
from .list_labels import str_labels_mimic as str_labels
from .list_labels import list_labels, translate_mimic_to_label
from .global_paths import jpg_path

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

class MIMICCXRDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.dataset_size = len(self.df)
        self.df[str_labels] = (self.df[str_labels].astype('float').abs().values > 0) * 1.
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        filepath = self.df.iloc[idx]["path"]
        img = imageio.imread(pre_process_path(filepath))
        mimic_gt = np.zeros([len(list_labels)])
        for et_label in translate_mimic_to_label:
            for destination_label in translate_mimic_to_label[et_label]:
                if self.df.iloc[idx][et_label]>0:
                    mimic_gt[list_labels.index(destination_label)]= 1.
        return img, mimic_gt