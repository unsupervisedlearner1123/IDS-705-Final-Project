# Analysis of Malware Prediction in Windows Devices <img width=90 align="right" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Duke_University_logo.svg/1024px-Duke_University_logo.svg.png">
## Final Project for IDS 705: Principles of Machine Learning 

*Contributors: Mohammad Anas, Ying Feng, Vicky Nomwesigwa, Deekshita Saikia*

### Abstract
Malware infection rates have been on the rise in recent years. Given the data deluge the industry currently faces, it is essential to guard our devices well so that the impact from malware attacks can be minimized. Through this analysis, we explore different tree-based models that can predict a Windows machine’s probability of getting infected by malware, based on different hardware and software configurations of that machine. A LightGBM classification model performed the best in our hold-out dataset in predicting if a machine would be infected. We also delve into which features of a device are most sensitive to a malware attack, and it was found that the diagonal display size is particularly sensitive to changes in the underlying sample.

### Data
The dataset used in this analysis was collected from [Kaggle](https://www.kaggle.com/competitions/microsoft-malware-prediction), provided by Microsoft to encourage open-source progress on effective techniques for predicting malware occurrences. The goal of this competition was to predict a Windows machine’s probability of getting infected by various families of malware, based on different properties of that machine. The telemetry data containing these properties and the machine infections was generated by combining heartbeat and threat reports collected by Microsoft's endpoint protection solution, Windows Defender.

The original dataset has 16 million values and 83 features, most of which were categorical attributes. Each row in this dataset corresponds to a machine, uniquely identified by a `MachineIdentifier`. `HasDetections` is the ground truth and indicates if malware was detected on the machine. A significant limitation of this analysis was the size of the datasets, which warranted significant compute and time resources. Owing to these constraints, we extract a stratified sample of approximately 1.5 million (~17%) observations from the original dataset. The stratification was performed on `HasDetections`, `Platform`, `Processor`, `IsProtected` and `Census_IsTouchEnabled`. The raw sample file can be found at `./data/OMG_OUR_LIFE_DEPENDS_ON_THIS.csv.zip`.

### Requirements
This project is built with Python 3, and the visualizations are created using Jupyter Notebooks. The following steps can be followed to replicate the analysis:

* Clone repo with SSH key
```
git clone git@github.com:unsupervisedlearner1123/IDS-705-Final-Project.git
```

* Install all packages for analysis
```
pip install -r requirements.txt
```

* Unzip files in `./data/` and set current working directory to repo's directory

* To replicate the data exporatory analysis, run code `10_Data_Preprocessing+EDA.ipynb`.

* To replicate the feature selection process, run code `20_Feature_Selection.ipynb`. The outut from code `10` feeds into code `20`.

* To replicate the pre-processing steps on unseen data, use code `preprocessing_pipeline.py`.

* To replicate the hyper-paramater tuning and model training steps, use code `30_tuning_LGBM.ipynb` to fit a LightGBM classifier, `31_tuning_RF.ipynb` to fit a Random Forest classifier, and `32_tuning_XGB.ipynb` to fit an XGBoost classifier.

* To test the classifiers on unseen data with tuned hyper-parameters, use code `40_Predictions_on_Test.py`.

* To replicate the experiments, use code `50_experiments.py`.
