# Analysis of Malware Prediction in Windows Devices

*Mohammad Anas, Ying Feng, Vicky Nomwesigwa, Deekshita Saikia*

### Project Objectives
Malware infection rates have been on the rise in recent years. Given the data deluge the industry currently faces, it is essential to guard our devices well so that the impact from malware attacks can be minimized. Through this analysis, we explore different tree-based models that can predict a Windows machine’s probability of getting infected by malware, based on different hardware and software configurations of that machine. A LightGBM classification model performed the best in our hold-out dataset in predicting if a machine would be infected. We also delve into which features of a device are most sensitive to a malware attack, and it was found that the diagonal display size is particularly sensitive to changes in the underlying sample.

### Data
The dataset used in this analysis was collected from [Kaggle](https://www.kaggle.com/competitions/microsoft-malware-prediction), provided by Microsoft to encourage open-source progress on effective techniques for predicting malware occurrences. The goal of this competition was to predict a Windows machine’s probability of getting infected by various families of malware, based on different properties of that machine. The telemetry data containing these properties and the machine infections was generated by combining heartbeat and threat reports collected by Microsoft's endpoint protection solution, Windows Defender.

The original dataset has 16 million values and 83 features, most of which were categorical attributes. Each row in this dataset corresponds to a machine, uniquely identified by a `MachineIdentifier`. `HasDetections` is the ground truth and indicates if malware was detected on the machine. A significant limitation of this analysis was the size of the datasets, which warranted significant compute and time resources. Owing to these constraints, we extract a stratified sample of approximately 1.5 million (~17%) observations from the original dataset. The stratification was performed on `HasDetections`, `Platform`, `Processor`, `IsProtected` and `Census_IsTouchEnabled`. The codebook can be referred to [here](https://www.kaggle.com/competitions/microsoft-malware-prediction/data?select=train.csv).

### Methods

### Experiments 

### Requirements
