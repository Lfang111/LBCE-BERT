
from sklearn.calibration import CalibratedClassifierCV as cc, calibration_curve
#from Bio import SeqIO
from pydpi.pypro import PyPro
import sys
import numpy as np
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, matthews_corrcoef, classification_report
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD as svd
from scipy import interp
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, SelectFromModel
from make_representations.sequencelist_representation import SequenceKmerRep, SequenceKmerEmbRep
#from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import make_scorer
import argparse
import pickle
import math
import xgboost as xgb
import pandas as pd


protein = PyPro()


def readAAP(file):  # read AAP features from the AAP textfile
    try:
        aapdic = {}
        aapdata = open(file, 'r')
        for l in aapdata.readlines():
            aapdic[l.split()[0]] = float(l.split()[1])
        aapdata.close()
        return aapdic
    except:
        print("Error in reading AAP feature file. Please make sure that the AAP file is correctly formatted")
        sys.exit()


def readAAT(file):  # read AAT features from the AAT textfile
    try:
        aatdic = {}
        aatdata = open(file, 'r')
        for l in aatdata.readlines():
            aatdic[l.split()[0][0:3]] = float(l.split()[1])
        aatdata.close()
        return aatdic
    except:
        print("Error in reading AAT feature file. Please make sure that the AAT file is correctly formatted")
        sys.exit()


def aap(pep, aapdic, avg):  # return AAP features for the peptides
    feature = []
    for a in pep:
        # print(a)
        if int(avg) == 0:
            score = []
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score.append(round(float(aapdic[a[i:i + 2]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score += float(aapdic[a[i:i + 2]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


def aat(pep, aatdic, avg):  # return AAT features for the peptides
    feature = []
    for a in pep:
        if int(avg) == 0:
            # print(a)
            score = []
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score.append(round(float(aatdic[a[i:i + 3]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score += float(aatdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            # print(a, score)
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


def CTD(pep):  # Chain-Transition-Ditribution feature
    feature = []
    name = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        ctd = protein.GetCTD()
        feature.append(list(ctd.values()))
        name = list(ctd.keys())
    return feature, name


def AAC(pep):  # Single Amino Acid Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        aac = protein.GetAAComp()
        feature.append(list(aac.values()))
        name = list(aac.keys())
    return feature, name


def DPC(pep):  # Dipeptide Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        dpc = protein.GetDPComp()
        feature.append(list(dpc.values()))
        name = list(dpc.keys())
    return feature, name


def PAAC(pep):
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        #paac=protein.GetMoranAuto()
        paac = protein.GetPAAC(lamda=4)
        feature.append(list(paac.values()))
        name = list(paac.keys())
    return feature, name


def kmer(pep, k):  # Calculate k-mer feature
    feature = SequenceKmerRep(pep, 'protein', k,norm='l1')
    return feature


def protvec(pep, k, file):  # Calculate ProtVec representation
    feature = SequenceKmerEmbRep(file, pep, 'protein', k)
    return feature

#def bertfea(file):
#    feature = []
#    with open(file, 'r') as f:
#        for line in f:
#            line = line.split(',')
            #feature.append([float(x) for x in line[1:201]])
#            feature.append([float(x) for x in line[1:769]])
    #pca = PCA(n_components=130)
    #pca = PCA(0.85)
    #pca.fit(feature)
     
    #pca_model = open('ibce_30_pca.pickle', 'wb')
    #pickle.dump(pca, pca_model)

    #feature=pca.transform(feature)
#    return feature


def FVs_fea(file):
    feature = []
    with open(file, 'r') as f:
        for line in f:
            line = line.split(',')
            #feature.append([float(x) for x in line[1:201]])
            feature.append([float(x) for x in line[1:]])
    return feature

def QSO(pep):
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        #paac=protein.GetMoranAuto()
        #qso = protein.GetQSO(maxlag=5)
        qso = protein.GetQSO(maxlag=10)
        feature.append(list(qso.values()))
        name = list(qso.keys())
    return feature, name

'''
def bertfea(file):
    feature = []
    import joblib
    idx_sorted = joblib.load('./shap/B-cell/virus/SARs-2/CLS.np')
    x_test = np.array(pd.read_csv(file,header=None,index_col=None,usecols=[i for i in range(1,769)]))
    return x_test[:, idx_sorted[:50]]
'''
def bertfea(file):
    feature = []
    with open(file, 'r') as f:
        for line in f:
            line = line.split(',')
            feature.append([float(x) for x in line[1:131]])
            #feature.append([float(x) for x in line[:]])
    return feature

def readpeptides(posfile, negfile):  # return the peptides from input peptide list file
    posdata = open(posfile, 'r')
    pos = []
    for l in posdata.readlines():
        if l[0] == '>':
            continue
        else:
            pos.append(l.strip('\t0\n'))
    posdata.close()
    negdata = open(negfile, 'r')
    neg = []
    for l in negdata.readlines():
        if l[0] == '>':
            continue
        else:
            neg.append(l.strip('\t0\n'))
    negdata.close()
    return pos, neg


def combinefeature(pep, featurelist, dataset):
    a=np.empty([len(pep), 1])
    fname=[]
    scaling = StandardScaler()
    #pca = svd(n_components=300)
    pca = PCA(0.99)
    vocab_name = []
    #pca = PCA(n_components=10)
    #print(a)
    if 'dpc' in featurelist:
        f_dpc, name = DPC(pep)
        # f_dpc = np.average(f_dpc, axis =1)
        a = np.column_stack((a, np.array(f_dpc)))
        fname = fname + name
    
    if 'paac' in featurelist:
        f_paac, name = PAAC(pep)
        #f_paac = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_paac)))
        fname = fname + name
    
    if 'kmer' in featurelist:
        kmers = kmer(pep, 2)
        #f_kmer = np.array(kmers.X.toarray())
        f_kmer = np.array(kmers.X.toarray())
        vocab_name = kmers.vocab

        a = np.column_stack((a, f_kmer))
        fname = fname + ['kmer']*len(f_kmer)
    
    if 'qso' in featurelist:
        f_qso, name = QSO(pep)
        #f_pa = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_qso)))
        fname = fname + name

    if 'ctd' in featurelist:
        f_ctd, name = CTD(pep)
        a = np.column_stack((a, np.array(f_ctd)))
        fname = fname + name 
    
    if 'protvec' in featurelist:
        # f_protvec = np.array(protvec(pep, 4, './protvec/sp_sequences_4mers_vec.bin').embeddingX)
        f_protvec = np.array(protvec(pep, 4, './protvec/sp_sequences_4mers_vec.txt').embeddingX)
        #f_protvec = pickle.load(open("features_protvec.pickle", 'rb'))
        #f_protvec = np.average(f_protvec, axis =1)
        a = np.column_stack((a, f_protvec))
        fname = fname + ['protvec']*len(f_protvec)

    if 'bertfea' in featurelist:
        #f_bertfea = np.array(bertfea('./bertfea/LBCEPred/train/CLS_fea.txt'))
        f_bertfea = np.array(bertfea('./shap/B-cell/ibce/tr_CLS.txt'))
        a = np.column_stack((a, f_bertfea))
        fname = fname + ['bertfea']*len(f_bertfea)
        
    if 'aac' in featurelist:
        f_aac, name = AAC(pep)
        a = np.column_stack((a, np.array(f_aac)))
        fname = fname + name

    if 'aat' in featurelist:
        #aatdic = readAAT("./aat/my_aat/all_lbce_data/aat_minmaxscaler_general.txt")
        aatdic = readAAT("./models/ibce/aat-general.txt.normal")
        f_aat = np.array([aat(pep, aatdic, 1)]).T
        a = np.column_stack((a, f_aat))
        #a = scaling.fit_transform(a)
        fname.append('AAT')

    if 'aap' in featurelist:
        #aapdic = readAAP("./aap/my_aap/all_lbce_data/aap_minmaxscaler_general.txt")
        aapdic = readAAP("./models/ibce/aap-general.txt.normal")
        f_aap = np.array([aap(pep, aapdic, 1)]).T
        a = np.column_stack((a,f_aap))
        #a = scaling.fit_transform(a)
        fname.append('AAP')

    if 'FVs_fea' in featurelist:
        #f_bertfea = np.array(FVs_fea('./FVs_fea/fea2/tr_fea.txt'))
        f_bertfea = np.array(FVs_fea('./FVs_fea/viral/train_fvs.txt'))
        a = np.column_stack((a, f_bertfea))
        fname = fname + ['FVs_fea']*len(f_bertfea)

    return a[:,1:], fname, vocab_name


def run_training(pos, neg, dataset):
    pep_combined = pos + neg
    pickle_info={}
    #print(pep_combined)
    # aap aat dpc aac kmer protvec paac qso ctd
    #featurelist = ['aac', 'paac', 'dpc', 'aat', 'bertfea']
    featurelist = ['aac', 'aap', 'aat', 'bertfea']
    print(featurelist)
    # featurelist = ['aac','aap','aat','protvec']
    pickle_info['featurelist'] = featurelist
    features, fname, vocab = combinefeature(pep_combined, featurelist, dataset) # 'aap', 'aat', 'aac' 
    print(len(features[0]))
    '''for i in range(len(features)):
    	print(features[i])'''
    pickle_info['feat_name'] = fname
    pickle_info['vocab'] = vocab
    #pickle.dump(features, open("features_latest.pickle", "wb"))
    #print(features)
    target = [1] * len(pos) + [0] * len(neg)
    #print(pep_combined)
    train(pep_combined, features, target, pickle_info, dataset)


def precision_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate prec for neg class
    '''
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return p

def recall_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate recall for neg class
    '''
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return r

def f1_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate f1 for neg class
    '''
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return f


def gridsearch(x, y, cv):
    scoring = {'AUPRC':'average_precision',
                'f1':'f1',
                'ACC':'accuracy',
                'prec':'precision',
                'recall':'recall',
                'AUROC':'roc_auc'}
    parameters = {
        #'max_depth':list(range(5, 15, 1)),
        'max_depth': [10],
        #'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate':[0.04],
        #'n_estimators':list(range(800, 2000, 100))
        'n_estimators': [1300]
    }
    xlf = xgb.XGBClassifier(max_depth=10,
                            learning_rate=0.01,
                            n_estimators=2000)
    optimized_GBM = GridSearchCV(xlf, param_grid=parameters, scoring=scoring, refit='AUROC', cv=cv, verbose=1, n_jobs=-1,
                                 return_train_score=True)
    optimized_GBM.fit(x, y)
    return optimized_GBM


def train(peptides, features, target, pickle_info, dataset):
    scaling = StandardScaler()
    scaling.fit(features)
    print(max(features[:,0]))
    x = scaling.transform(features)
    #print(max(x[:,1]))
    y = np.array(target)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
    model = gridsearch(x, y, cv)
    #aapdic = readAAP("./training/"+dataset+"/aap-general.txt.normal")
    aapdic = readAAP("./models/ibce/aap-general.txt.normal")
    #aatdic = readAAT("./training/"+dataset+"/aat-general.txt.normal")
    aatdic = readAAT("./models/ibce/aat-general.txt.normal")
    pickle_info ['aap'] = aapdic
    pickle_info ['aat'] = aatdic
    pickle_info ['scaling'] = scaling
    pickle_info ['model'] = model
    pickle_info ['training_features'] = features
    pickle_info ['training_targets'] = y
    #pickle.dump(pickle_info, open("./result/ibce/xgb-ibce-paac1-aap-aat-CLS130.pickle", "wb"))
    print("Best parameters: ", model.best_params_)
    print("Best AUC_ROC:", model.best_score_)
    predict = model.best_estimator_.predict_proba(x)
    data = pd.DataFrame(model.cv_results_)
    data.T.to_csv('all_fold_data.csv')
    #ff = open('predict.txt', 'w')
    #ff.write(str([i for i in predict]))
    #ff.write('\n')
    #ff.write(str([i for i in y]))
    #print(predict[:, 1])
    #newdf = pd.DataFrame({'aap':predict[:, 1]})
    #newdf.to_csv('aap.csv')
    #results = model.cv_results_
    #bi = model.best_index_
    #print("roc_auc:",results['mean_test_auc_score'][bi],
    #      "accuracy:",results['mean_test_accuracy'][bi],
    #      "precision +:",results['mean_test_scores_p_1'][bi],
    #      "recall +:",results['mean_test_scores_r_1'][bi],
    #      "f1 +:",results['mean_test_scores_f_1_1'][bi],
    #      "precision -:",results['mean_test_scores_p_0'][bi],
    #      "recall -:",results['mean_test_scores_r_0'][bi],
    #      "f1 -:",results['mean_test_scores_f_1_0'][bi],
    #      "precision_micro:",results['mean_test_precision_micro'][bi],
    #      "f1 -:",results['mean_test_precision_macro'][bi],
    #      "mcc -:",results['mean_test_mcc'][bi])

    cv_accracy = model.cv_results_['mean_test_ACC'][model.best_index_]
    cv_auprc = model.cv_results_['mean_test_AUPRC'][model.best_index_]
    cv_precision = model.cv_results_['mean_test_prec'][model.best_index_]
    cv_recall = model.cv_results_['mean_test_recall'][model.best_index_]
    cv_auroc = model.cv_results_['mean_test_AUROC'][model.best_index_]
    cv_f1 = model.cv_results_['mean_test_f1'][model.best_index_]

    y_train_t=y.tolist()
    y_train_t.count(1)
    y_train_t.count(0)
    TP1=y_train_t.count(1)*cv_recall
    FP1=(TP1/cv_precision)-TP1
    TN1=y_train_t.count(0)-FP1
    FN1=y_train_t.count(1)-TP1

    print('TP:',TP1,',TN:',TN1,',FP:',FP1,',FN:',FN1)
    cv_specificity = Specificity=TN1/(TN1+FP1)

    if ((float(TP1 + FP1) * float(TN1 + FN1)) != 0):
        cv_MCC = float(TP1*TN1-FP1*FN1)/ math.sqrt(float(TP1 + FP1) * float(TP1 + FN1) * float(TN1 + FP1) * float(TN1 + FN1))
        print('Specificity_train:',cv_specificity,',ACC_train:',cv_accracy,',Precision_train:',cv_precision,',Recall_train:',cv_recall,',F1Score_train:',cv_f1,',MCC_train:',cv_MCC,',auprc_train:',cv_auprc,',auroc_train:',cv_auroc)
    else:
        print('Specificity_train,ACC_train,Precision_train,Recall_train,F1Score_train,auprc_train,auroc_train:',
              cv_specificity,cv_accracy,cv_precision,cv_recall,cv_f1,cv_auprc,cv_auroc)


if __name__ == "__main__":
    dataset = sys.argv[1]
    pos, neg = readpeptides("./datasets/training/"+dataset+"/pos.txt",
                            "./datasets/training/"+dataset+"/neg.txt")
    #print(pos, neg)
    run_training(pos, neg, dataset)
