# a list of the labels in the mimic-cxr dataset, used for selecting the label columns
# in the mimic-cxr label table
str_labels_mimic = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']

# list of labels reported in the paper. the order of these labels is important
# when modifying the code, since it stablishes the index of each label in any tensor
# that has a labels channel
list_labels = ['Edema',
'Opacity',
'Pneumothorax',
'Pleural abnormality',
'Lung Lesion',
'Atelectasis',
'Consolidation',
'Fracture',
'Cardiomegaly',
'Abnormal Mediastinum Contour']

# dictionary used to translate from the labels of the mimic-cxr dataset to
# the labels reported in the paper
translate_mimic_to_label = {'Edema':['Edema','Opacity'], 
'Pneumothorax':['Pneumothorax'], 
'Pleural Other':['Pleural abnormality'],
'Pleural Effusion':['Pleural abnormality'],
'Lung Lesion':['Lung Lesion', 'Opacity'],
'Atelectasis':['Atelectasis', 'Opacity'],
'Consolidation':['Consolidation', 'Opacity'],
'Fracture':['Fracture'],
'Lung Opacity':['Opacity'],
'Pneumonia': ['Opacity'],
'Cardiomegaly':['Cardiomegaly'],
'Enlarged Cardiomediastinum':['Abnormal Mediastinum Contour']}

# dictionary used to translate from labels of the reflacx dataset
# and labels from the modified version of the chexpert-labeler
# to the labels that were reported in the paper
translate_et_to_label = {'Pulmonary edema':['Edema','Opacity'], 
'Pneumothorax':['Pneumothorax'], 
'Pleural abnormality':['Pleural abnormality'],
'Pleural thickening':['Pleural abnormality'],
'Pleural effusion':['Pleural abnormality'],
'Lung nodule or mass':['Lung Lesion', 'Opacity'],
'Nodule':['Lung Lesion', 'Opacity'],
'Mass':['Lung Lesion', 'Opacity'],
'Atelectasis':['Atelectasis', 'Opacity'],
'Consolidation':['Consolidation', 'Opacity'],
'Acute fracture':['Fracture'],
'Fracture':['Fracture'],
'Groundglass opacity':['Opacity'],
'Interstitial lung disease': ['Opacity'],
'Fibrosis': ['Opacity'],
'Enlarged cardiac silhouette':['Cardiomegaly'],
'Abnormal mediastinal contour':['Abnormal Mediastinum Contour'],
'Wide mediastinum':['Abnormal Mediastinum Contour'],
'wide mediastinum & abnormal mediastinum contour':['Abnormal Mediastinum Contour'],
'interstitial lung disease & fibrosis':['Opacity'],
'acute fracture & fracture':['Fracture']}