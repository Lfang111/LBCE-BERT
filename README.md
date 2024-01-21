# LBCE-BERT

<h3>Folder Structure</h3>
├── train.py - the model is trained by cross-validation<br/>
├── predict.py - predict unknown linear BCE<br/>
├── datasets/ - source data<br/>
│&emsp; └── loader.py<br/>
│ <br/>
├── models/ - cross-validate model<br/> 
│&emsp;          aap, aat antigenicity scale<br/>
│ <br/>
├── pydpi/ - functions for feature calculations<br/>
│&emsp;          (Instructions for use are detailed in pydpi/manual/UserGuide.pdf)<br/>
│ <br/>
│ <br/>
└── utils/ - small utility functions<br/>
    ├── util.py<br/>
    └── logger.py - set log dir for tensorboard and logging output<br/>

<h3>Training & Predict</h3>
    
1. Extracting BERT embedding

BERT models and codes are taken from https://github.com/BioSequenceAnalysis/Bert-Protein.
Here the pre-trained BERT model using one amino acid residue as a word.

2. Training model

python3 xgb_retrain.py 

3. Predict

python3 predict.py
