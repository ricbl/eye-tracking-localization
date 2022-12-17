# Localization supervision of chest x-ray classifiers using label-specific eye-tracking annotation

This repository contains code for the paper ["Localization supervision of chest x-ray classifiers using label-specific eye-tracking annotation"](https://arxiv.org/abs/2207.09771). This paper proposes a procedure for training deep learning models using eye-tracking data. This procedure uses timestamps from CXR reports and eye-tracking data to extract label-specific localization information. The use of this information then improves the interpretability of the tested models, as measured by the task of abnormality localization. The eye-tracking data and CXR reports were sourced from the [REFLACX dataset](https://www.physionet.org/content/reflacx-xray-localization/1.0.0/), and the CXR images were sourced from the [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

## Prerequisites

1. For running scripts from the `src/chexpert-labeler/` folder, follow the [Prerequisites instructions from the chexpert-labeler](https://github.com/stanfordmlgroup/chexpert-labeler/tree/4629609647d027b1dc9d4f340f5d3c03b4fb4e4f#prerequisites).

1. For running scripts from the `src/` folder, install the following python libraries:
- python==3.9.7
- h5py==2.10.0
- imageio==2.9.0
- joblib==1.1.0
- matplotlib==3.5.0
- numpy==1.21.2
- pandas==1.3.5
- pillow==8.4.0
- scikit-image==0.18.3
- scikit-learn==1.0.2
- scipy==1.6.2
- tensorboardX==2.2
- pydicom==2.3.0
- torch==1.10.2
- torchvision==0.11.3
- opencv==4.5.1

3. Install the library containing the class that stores datasets into HDF5 files: 
```
    cd hdf5_library
    pip install -e .
    cd ..
```

4. Put the REFLACX dataset's `main_data` folder in the root of this repository. Put MIMIC-CXR-JPG dataset's `files` folder in a folder named `mimic_images`, and the tables `mimic-cxr-2.0.0-chexpert.csv` and `mimic-cxr-2.0.0-split.csv` in a folder named `mimic_tables`. You may also choose other locations for the datasets by changing the paths defined in the `src/global_paths.py` file.

5. To generate the labels of the reports from the REFLACX dataset using the modified chexpert-labeler, run 

    `python -m src.extract_report`,

    followed by 

    `python src/chexpert-labeler/label.py --reports_path=phase_2.csv --output_path=labeled_reports_2.csv`
	
	and 
	
	`python src/chexpert-labeler/label.py --reports_path=phase_3.csv --output_path=labeled_reports_3.csv`.

    The modified rules, implemented with the help of a cardiothoracic subspecialty-trained radiologist, can be found in `src/chexpert-labeler/phrases/mention/`, `src/chexpert-labeler/phrases/unmention/`, and `src/chexpert-labeler/patterns/negation.txt`. 

6. To generate heatmaps for each sentence in the REFLACX dataset, run 

    `python -m src.find_fixations_all_sentences`. 

7. To generate a preprocessed list containing information about the CXRs to be used from the MIMIC-CXR-JPG dataset, run 

    `python -m src.mimic_generate_df`.

## Training

To train each of the models from the paper, use:
- for the Unannotated model:

    `python -m src.train --gpus=0 --experiment=unannotated_baseline --last_layer 3 4 --grid_size=32 --use_grid_balancing_loss=true --weight_decay=1e-5 --unet=true --gamma_unet=300 --weight_loss_annotated=3 --dataset_type=u`;

- for the Ellipses model: 

    `python -m src.train --gpus=0 --experiment=ellipses_baseline --last_layer 3 4 --grid_size=32 --use_grid_balancing_loss=true --weight_decay=1e-5 --unet=true --gamma_unet=300 --weight_loss_annotated=3 --use_et=False`

(for training the models using only 50% or 25% of the dataset annotated with ellipses, add ` --percentage_annotated=0.5` or ` --percentage_annotated=0.25` to the command above);

- for the ET model (ours): 

    `python -m src.train --gpus=0 --experiment=et_data_model --last_layer 3 4 --grid_size=32 --use_grid_balancing_loss=true --weight_decay=1e-5 --unet=true --gamma_unet=300 --weight_loss_annotated=3 --use_et=True`.

## Evaluating

For getting the AUC and IoU values as reported in the paper, run two validation runs for each of the training experiments, one for the validation split and one for the test split:

- `python -m src.train --skip_train=true --nepochs=1 --load_checkpoint_d=<path to experiment folder>/state_dict_d_best_epoch --batch_size=8 --last_layer 3 4 --grid_size=32 --unet=True --num_workers=4 --experiment=<experiment name> --gpus=0 --calculate_cam=true --use_grid_balancing_loss=true --split_validation=val`

- `python -m src.train --skip_train=true --nepochs=1 --load_checkpoint_d=<path to experiment folder>/state_dict_d_best_epoch --batch_size=8 --last_layer 3 4 --grid_size=32 --unet=True --num_workers=4 --experiment=<experiment name> --gpus=0 --calculate_cam=true --use_grid_balancing_loss=true --split_validation=test`

Then, put all folders from these evaluation runs in split-separated folders (`val/` and `test/`) and run 

`python -m src.get_auc_iou_tables --val_folder=<path to the val folder> --test_folder=<path to the test folder>`.

Result tables, formatted as used in LaTeX, will be written to the files `Table2.txt`, `Table3.txt`, and `Table4.txt`.

The average AUC and IoU results are:

| Metric      | Unannotated | Ellipses | ET model (ours) |
| --- | --- | --- | --- |
| AUC | 0.767 | 0.765 | 0.765 | 
| IoU | 0.201 | 0.335 | 0.256 |

Numbers for the other tables from the paper can be shown in the command line using:
- Table 1: `python -m src.evaluate_chexpert`;
- Table S2: `python -m src.eyetracking_dataset` and `python -m src.mimic_dataset`.

### Ablation

We present below how to get the numbers of the rows of the ablation study table that have at least one part removed from the method:

| Row      | Label Specific Heatmap | Balanced Range Normalization | Multi-Resolution Architecture | Multi-Task Learning |
| --- | --- | --- | --- | --- |
| 1 | &#9744; | &#9744; | &#9744; | &#9744; | &#9744; | 
| 2 | &#9745; | &#9744; | &#9744; | &#9744; |
| 3 | &#9745; | &#9745; | &#9744; | &#9744; |
| 4 | &#9745; | &#9745; | &#9745; | &#9744; |
| 5 | &#9744; | &#9745; | &#9745; | &#9745; |

Training commands:
- Row 1: `python -m src.train --experiment=ablation_row_1 --gpus=0 --last_layer 4 --grid_size=16 --use_grid_balancing_loss=false --weight_decay=1e-5 --unet=false --gamma_unet=300 --weight_loss_annotated=3 --use_et=True --calculate_label_specific_heatmaps=False`
- Row 2: `python -m src.train --gpus=0 --experiment=ablation_row_2 --last_layer 4 --grid_size=16 --use_grid_balancing_loss=false --weight_decay=1e-5 --unet=false --gamma_unet=300 --weight_loss_annotated=3 --use_et=True`
- Row 3: `python -m src.train --experiment=ablation_row_3 --gpus=0 --last_layer 4 --grid_size=16 --use_grid_balancing_loss=true --weight_decay=1e-5 --unet=false --gamma_unet=300 --weight_loss_annotated=3 --use_et=True`
- Row 4: `python -m src.train --experiment=ablation_row_4 --gpus=0 --last_layer 3 4 --grid_size=32 --use_grid_balancing_loss=true --weight_decay=1e-5 --unet=false --gamma_unet=300 --weight_loss_annotated=3 --use_et=True`
- Row 5: `python -m src.train --experiment=ablation_row_5 --gpus=0 --last_layer 3 4 --grid_size=32 --use_grid_balancing_loss=true --weight_decay=1e-5 --unet=true --gamma_unet=300 --weight_loss_annotated=3 --use_et=True --calculate_label_specific_heatmaps=false`

Validation/testing commands: make the same changes as done for the training commands.

## Images

The images used in Figure 1 are generated in the root of this repository when running `python -m src.find_fixations_all_sentences.py`, from the Prerequisites. The images from Figures 2 and S1 were generated by running `./get_paper_images.sh`. When running these commands, images are saved to the experiment output folder.
