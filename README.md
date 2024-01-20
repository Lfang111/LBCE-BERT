# LBCE-BERT

<h3>Folder Structure</h3>
├── train.py - the model is trained by cross-validation<br/>
├── predict.py - predict unknown linear BCE<br/>
├── data/ - anything about data loading goes here
│   └── loader.py
│
├── model/ - pre-trained content encoder model
│
├── data/ - default directory for storing experimental datasets
│
├── model/ - networks, models and losses
│   ├── encoder.py
│   ├── gmm.py
│   ├── loss.py
│   ├── model.py
│   └── transformer.py
│
│  
└── utils/ - small utility functions
    ├── util.py
    └── logger.py - set log dir for tensorboard and logging output

<h3>Training & Predict</h3>
    
1. Extracting BERT embedding

BERT models and codes are taken from https://github.com/BioSequenceAnalysis/Bert-Protein.
Here the pre-trained BERT model using one amino acid residue as a word.

2. Training model

python3 xgb_retrain.py 

3. Predict

python3 predict.py
