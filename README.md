
# MF2N: Multiview feature fusion network for pancreatic cancer segmentation
The GPU should be at least 10GB. Using pytorch, cuDNN 8.0.2 or newer. At least 6 CPU cores (12 threads) are recommended.

We present a novel deep network, MF2N network, equipped with three feature fusion modules in different views for automated segmentation of the variable pancreas and tumors, which is challenging due to 1) large variations occurring within/crossing the narrow and irregular pancreas, 2) a small proportion of voxels on pancreatic tumors in the abdomen and 3) ambiguous lesion boundaries caused by low contrast between target surroundings. We first propose a novel adaptive morphological feature fusion (AMF2) module to dynamically learn and fuse morphological features of the pancreas and tumors from the skeleton to boundaries, aiming to mitigate undersegmentation of targets. Then, bidirectional semantic feature fusion (BSF2) module is proposed to optimize mutual information between prediction and manual delineation as well as discard redundant information between input and attentive features to capture more consistent feature expressions and alleviate noise interference caused by the large fraction of background in the abdomen. Furthermore, we develop a local-global dependency feature fusion (LGDF2) module module to fuse local features from CNNs and global information provided by shallow features through a lightweight transformer to enhance MF2N’s capability to grasp more boundary and content features.

# Environment configuration
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
``` 
Modify the .bashrc file according to your path.
```bash
cd /root/XX
source /root/XX/.bashrc
``` 

# Usage
### Experiment planning and preprocessing
Using devices for data processing
```bash
export CUDA_VISIBLE_DEVICES=0
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```
`XXX` is task name `TaskXXX_MYTASK`.
Here we utilize nnUNet_plan_and_preprocess -t 007 to process the pancreas dataset, and subsequently generate preprocessed data in the directory nnUNet_preprocessed/TaskXXX_MYTASK. These files include the generated configurations, which will be read by the trainer. Please note that the preprocessed data folder only contains the training cases. The test images are not preprocessed.

### Model training
Carry out fivefold cross-validation experiments:
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD 
```
The string “CONFIGURATION” is used to identify the requested U-Net configuration. TRAINER_CLASS_NAME is the name of the model trainer. TASK_NAME_OR_ID specifies which datasets should be used for training, and FOLD specifies which fold of the five-fold cross-validation should be used. We are using nnUNet_train 3d_lowres nnUNetTrainerV2PanRegions 7 fold, where fold can be 0, 1, 2, 3, or 4. If you want to continue training, simply add “-c” to the command.

Finally, the trained model is written to the RESULTS_FOLDER/nnUNet folder

For Task007_Pancreas:

    RESULTS_FOLDER/nnUNet/
    ├──  3d_lowres
    │   └── Task07_Pancreas
    │       └── nnUNetTrainerV2PanRegions
    │           ├── fold_0
    │           │   ├── debug.json
    │           │   ├── model_final_checkpoint.model
    │           │   ├── model_final_checkpoint.model.pkl
    │           ├── fold_1
    │           ├── fold_2
    │           ├── fold_3
    │           └── fold_4
    └── 

- debug.json: Contains blueprint and inferred parameters for training the model.
- model_final_checkpoint.model / model_final_checkpoint.model.pkl: checkpoint file of the final model (after training). This is used for testing.

### Run inference

```bash
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION 
```
After the training, we used the command "nnUNet_predict -i /public/home/../ -o /public/home/../predict -tr nnUNetTrainerV2BraTSRegions_DA3 -m 3d_lowres -p nnUNetPlansv2.1 -t Task007_Pancreas" to inference.
