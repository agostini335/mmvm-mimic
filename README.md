# MMVM VAE on MIMIC-CXR

![diagram-20241030 (1)](https://github.com/user-attachments/assets/d563f218-fe57-4f0f-b781-b1d6075fc178)

This repository provides the code needed to reproduce the MIMIC-CXR experiment from our NeurIPS 2024 paper:  
**"Unity by Diversity: Improved Representation Learning in Multimodal VAEs"**.

The main repository, which includes code for replicating additional experiments from the paper,
can be found at the following link: https://github.com/thomassutter/mmvmvae.

We welcome any comments or questions — please feel free to reach out!



# Instructions to Run the Code

## Environment

1. **Set up the Environment**
   - Use the `mimic_environment.yml` file to create the required Conda environment.

## Data Preparation

### Step 1: Data Download and Preprocessing

1. **Download data**
   - The data for the MIMIC-CXR experiment can be downloaded though the link: https://physionet.org/content/mimic-cxr-jpg/2.1.0/.

2. **Configure and Preprocess the Dataset**
   - Adjust parameters in `config/preprocessing_config.yaml` as needed.
   - Run `prepare_dataset.py` for each `view_position` [AP/PA/Lateral/LL].
   - **Output:** A folder for each view position with preprocessed images (NumPy arrays) and metadata.

### Step 2: Building Multimodal Dataset Splits

1. **Configure Dataset Settings**
   - Adjust parameters in `config/DatasetConfig.py`as needed. The `dir_data` folder must contain both `mimic-cxr-2.0.0-metadata.csv` and `mimic-cxr-2.0.0-chexpert.csv` original files.

2. **Generate Dataset Splits**
   - Run the notebook `notebooks/data_points_generator.ipynb` to create an indexing file with the `all_combi_no_missing` policy. The `path` parameter must be set to dir_data.
   - Run the script `generate_cache.py` to generate train-val-test splits.
   - **Output:** Dataset splits saved to the `cache` folder.

### Step 3: Convert Dataset to Dask Format
1. **Convert to Dask Format**
   - Run the notebook `notebooks/from_npy_to_dask.ipynb` and execute the conversion steps. The `out_path` parameter must be set to `dir_data`. 
   - **Output:** Dask-formatted dataset.

## Experiments

### Train VAE Models
   - Adjust parameters in `config/ModelConfig.py` and `config/MyMVWSLConfig.py`
   - Run the following script:
     ```bash
     python main_mv_wsl.py
     ```
### Train Supervised Classifiers
   - Adjust parameters in `MyClfConfig.py`
   - Run the following script:
     ```bash
     python main_train_clf_MimicCXR.py
     ```
## Citation
If you use our model in your work, please cite us using the following citation

```
@inproceedings{sutter2024,
  title={Unity by Diversity: Improved Representation Learning in Multimodal VAEs},
  author={Sutter, Thomas M and Meng, Yang and Agostini, Andrea and Chopard, Daphné and Fortin, Norbert and Vogt, Julia E. and Shahbaba, Babak and Mandt, Stephan},
  year = {2024},
  booktitle = {arxiv},
}
```
---

