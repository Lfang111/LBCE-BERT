# LBCE-BERT

<h3>Folder Structure</h3>
├── train.py - the model is trained by cross-validation<br/>
├── predict.py - predict unknown linear BCE<br/>
├── bertfea/ - BERT embedding<br/> 
│&emsp;├── ABCPred/<br/>
│&emsp;├── BCPreds/<br/>
│&emsp;├── Blind387/<br/>
│&emsp;├── Chen/<br/>
│&emsp;└── iBCE-EL_independent/<br/>
│ <br/>
├── datasets/ - source data<br/>
│&emsp;├── ABCPred/<br/>
│&emsp;├── BCPreds/<br/>
│&emsp;├── Blind387/<br/>
│&emsp;├── Chen/<br/>
│&emsp;├── LBtope/<br/>
│&emsp;├── iBCE-EL_independent/<br/>
│&emsp;├── iBCE-EL_training/<br/>
│&emsp;└── training<br/>
│ <br/>
├── models/ - cross-validate model; aap, aat antigenicity scale<br/> 
│ <br/>
├── pydpi/ - functions for feature calculations (Instructions for use are detailed in pydpi/manual/UserGuide.pdf)<br/>
│ <br/>
│ <br/>
└── pydpi/ - functions for feature calculations (Instructions for use are detailed in pydpi/manual/UserGuide.pdf)<br/>

<h3>Training & Predict</h3>
    
1. Extracting BERT embedding

BERT models and codes are taken from https://github.com/BioSequenceAnalysis/Bert-Protein.<br/>
Here the pre-trained BERT model using one amino acid residue as a word.

2. Training model

python3 xgb_retrain.py 

3. Predict

python3 predict.py
