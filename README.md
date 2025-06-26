# SIMPLE
SIMPLE is a simultaneous multi-plane self-supervised learning method for isotropic MRI restoration from anisotropic data.

![image](figures/model_arch_with_loss.png)

link for arxiv paper: https://www.arxiv.org/abs/2408.13065

## Installation

To use this project, use the following steps:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/TechnionComputationalMRILab/SIMPLE.git
2. **Install Dependencies (via conda)**
   ```sh
   conda env create -f environment.yml
   
## Pre-Processing
You should provide a csv file that contains a table with 2 or 3 columns.
Each column represents the path for MRI case (DICOM/Nifti format) in specific plane. The column name can be 'coronal'/'axial'/'sagittal'.
Each row represents the cases for specific patient.
The csv file should be located under SIMPLE directory.

For example:

![image](figures/csv_file_example.png)
## Training
This project contains 2 models: ATME and SIMPLE.

ATME is used as a preliminary stage for creating HR MRI images. 
In order to train SIMPLE, you should train first 2 or 3 ATME models for the coronal, axial and sagittal planes separately and then evaluating each of them on the whole dataset.

For both models you must specify the following base flags: 

--isTrain

--main_root (main directory name for all models outputs)

--model_root (directory name for model outputs)

--eval_plane (evaluation plane)

--csv_name (csv file name)

--data_format (dicom/nifti)

--global_min / --global_max (specify the grayscale range of the images, else the code calculate the absolute minimum and maximum values)

--phase (train/test)

--vol_cube_dim (the dimension of the resulted cube MRI volume - can be any value above 256). *Pay attention that different cube dimension may require different number of discriminator layers (--n_layers_D) for ATME training (for example: 256 require 3 discriminator layers, 512 require 4 discriminator layers)

--calculate_dataset or --no-calculate_dataset (whether to perform pre-processing for the dataset or not. Data pre-processing must be done before the training).

*For more flags, please see 'options' directory.

- For training ATME, run 'train.py atme' command with the base flags and specify also the following flags: --plane (coronal/axial/sagittal), --TestAfterTrain (whether to perform evaluation to the ATME model immediately after its training).

   *In order to evaluate the ATME model separately you can add the flag --no-isTrain instead the flag --isTrain to train.py script or to use the test.py script (see evaluation section).

   **You can train SIMPLE model only after evaluating ATME model in each plane.

   Example:

   ```sh
   python train.py atme --isTrain --eval_plane=coronal --plane=axial --main_root=outputs --model_root=atme_axial_output --csv_name=<file_name>.csv --vol_cube_dim=512 --data_format=nifti --calculate_dataset 
   ```

- For training SIMPLE, run 'train.py simple' command with the base flags and specify also the flags --planes (specify which planes the model is based on), --atme_cor_root/--atme_ax_root/--atme_sag_root (the coronal/axial/sagittal ATME outputs directory).

  Example:

   ```sh
   python train.py simple --isTrain --eval_plane=coronal --planes=coronal,axial,sagittal --main_root=outputs --model_root=simple_output --csv_name=<file_name>.csv --vol_cube_dim=512 --calculate_dataset --atme_cor_root=atme_coronal_output --atme_ax_root=atme_axial_output --atme_sag_root=atme_sagittal_output --data_format=nifti 
   ```

## Evaluation
- for evaluating ATME, run 'test.py atme' command with the base flags and specify also the flag --plane (coronal/axial/sagittal).

  Example:

   ```sh
   python test.py atme --eval_plane=coronal --plane=axial --main_root=outputs --model_root=atme_axial_output --csv_name=<file_name>.csv --vol_cube_dim=512 --data_format=nifti
   ```
- for evaluating SIMPLE, run 'test.py simple' command with the base flags and specify also the flag --planes (specify which planes the model is based on), --atme_cor_root/--atme_ax_root/--atme_sag_root (the coronal/axial/sagittal ATME outputs directory).
   ```sh
   python test.py simple --eval_plane=coronal --planes=coronal,axial,sagittal --main_root=outputs --model_root=simple_output --csv_name=<file_name>.csv --vol_cube_dim=512 --data_format=nifti
   ```

## Pre-Trained Model
For evaluating the model (ATME/SIMPLE) using pre-trained model, please add the flag --pre_train_G_path to specify the pre-trained generator path.

For evaluating ATME using pre-trained model, please add also the flag --pre_train_W_path to specify the pre-trained W network path.

## Contact

Please contact us on be.rotem@campus.technion.ac.il

## References
- Edgardo Solano-Carrillo, Angel Bueno Rodriguez, Borja Carrillo-Perez, Yannik Steiniger, and Jannis Stoppe. Look atme: the discriminator mean entropy needs attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 787–796, 2023
- Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.
