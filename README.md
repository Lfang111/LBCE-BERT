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
├── datasets/ - source data<br/>
│&emsp;&emsp;├── ABCPred/<br/>
│&emsp;&emsp;├── BCPreds/<br/>
│&emsp;&emsp;├── Blind387/<br/>
│&emsp;&emsp;├── Chen/<br/>
│&emsp;&emsp;├── LBtope/<br/>
│&emsp;&emsp;├── iBCE-EL_independent/<br/>
│&emsp;&emsp;├── iBCE-EL_training/<br/>
│&emsp;&emsp;├── training<br/>
│&emsp;&emsp;├── fasttext_generated_ngram_supervised.py<br/>
│&emsp;&emsp;└── shuffle_data_1.py<br/>
├── models/ - cross-validate model; aap, aat antigenicity scale<br/> 
└── pydpi/ - functions for feature calculations (Instructions for use are detailed in pydpi/manual/UserGuide.pdf)<br/>

<h3>Training & Predict</h3>
    
<h4>1. Extracting BERT embedding</h4>

BERT models and codes are taken from https://github.com/BioSequenceAnalysis/Bert-Protein.<br/>
Here the pre-trained BERT model using one amino acid residue as a word.

<h4>2. Training model</h4>

python3 xgb_retrain.py  file_path </br>

<strong>Note:</strong>  Modify the BERT embedding file at line 224.</br>
&emsp;&emsp;&emsp;Read training data positive and negative samples to be two files.</br>

<h4>3. Predict</h4>

python3 predict.py  file
