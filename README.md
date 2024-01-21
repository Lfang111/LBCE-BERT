# LBCE-BERT

<h3>Folder Structure</h3>
├── train.py - the model is trained by cross-validation<br/>
├── predict.py - predict unknown linear BCE<br/>
├── bertfea/ - BERT embedding<br/> 
│&emsp;&emsp;├── ABCPred/<br/>
│&emsp;&emsp;├── BCPreds/<br/>
│&emsp;&emsp;├── Blind387/<br/>
│&emsp;&emsp;├── Chen/<br/>
│&emsp;&emsp;└── iBCE-EL_independent/<br/>
│&emsp; <br/>
├── datasets/ - source data<br/>
│&emsp;&emsp;├── ABCPred/<br/>
│&emsp;&emsp;├── BCPreds/<br/>
│&emsp;&emsp;├── Blind387/<br/>
│&emsp;&emsp;├── Chen/<br/>
│&emsp;&emsp;├── LBtope/<br/>
│&emsp;&emsp;├── iBCE-EL_independent/<br/>
│&emsp;&emsp;├── iBCE-EL_training/<br/>
│&emsp;&emsp;└── training<br/>
│&emsp; <br/>
├── models/ - cross-validate model; aap, aat antigenicity scale<br/> 
│&emsp; <br/>
└── pydpi/ - functions for feature calculations (Instructions for use are detailed in pydpi/manual/UserGuide.pdf)<br/>

<h3>Training & Predict</h3>
    
1. Extracting BERT embedding

BERT models and codes are taken from https://github.com/BioSequenceAnalysis/Bert-Protein.<br/>
Here the pre-trained BERT model using one amino acid residue as a word.

2. Training model

python3 xgb_retrain.py 

3. Predict

python3 predict.py
