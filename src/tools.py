'''basics'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import Counter
from typing import Iterable, Tuple

'''features'''
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize

'''Classifiers'''
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import class_weight


'''Metrics/Evaluation'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from scipy import interp
from itertools import cycle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

import operator    
import joblib


def show_labels(y_true:Iterable, y_hat:Iterable, title:str = 'Classifier', class_range:Tuple[int,int] = (1,16)):
    """
    Plot heatmap of confusion matrix for SDGs.
    
    Parameters
    ----------
    y_true : array-like
            The input array of true labels.
    y_hat : array-like
            The input array of predicted labels.
    title : str, default 'Classifier'
            A title of the plot to be displayed.
    class_range : Tuple[int,int], default (1,18)
            A tuple of SDG range. The default value assumes that SDGs 1 through 17 are used.
            If some SGDs are missing, adjust class_range accordingly.
            
     Returns
    -------
    Has not return statement. Prints a plot.
    
    """
    assert len(y_true) == len(y_hat), "Arrays must be of the same length"
    
    to_labels = list(range(class_range[0],class_range[1]))
    to_accuracy = accuracy_score(y_true, y_hat)
    
    df_lambda = pd.DataFrame(confusion_matrix(y_true, y_hat),
                             index = list(range(class_range[0], class_range[1])),
                             columns = list(range(class_range[0], class_range[1]))
                            )
    
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_lambda, annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_ylim(ax.get_ylim()[0] + 0.5, ax.get_ylim()[1] - 0.5)
    
    plt.title(title + f'\nAccuracy:{round(to_accuracy, 3)}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label');
    
def get_topwords(logit_model, vectorizer, n_models:int = 15, n:int = 30, show_idxmax:bool = True) -> pd.DataFrame:
    """
    Extract top n predictors with highest coefficients from a logistic regression model and vectorizer object.
    
    Parameters
    ----------
    logit_model : LogisticRegression estimator
            A fitted LogisticRegression object from scikit-learn with coef_ attribute
    vectoriser : CountVectorizer or TfidfVectorizer
            A fitted CountVectorizer or TfidfVectorizer object from scikit-learn with get_feature_names attribute.
    n_models : int, default 17
            Indicates the number of models fitter by logit_model, i.e. n_classes.
    n : int, default 30
            The number of top predictors for each model to be returned. If None, returns all predictors
    show_idxmax : bool default True
            Indicates whether to print the keyword/predictor for each class
    Returns
    -------
    df_lambda : a pandas DataFrame object of shape (n_models,1) with a columns Keywords. Each cell in the column is
    a sorted list of tupples with top n predictors which has the form of (keyword, coefficient).
    
    """
    
    
    df_lambda = pd.DataFrame(logit_model.coef_,
                         columns = vectorizer.get_feature_names_out(),
                         index = [f'sdg_{x}' for x in range(1,n_models+1)]).round(3)
    
    if show_idxmax:
        display(df_lambda.idxmax(axis = 1))
    
    df_lambda = pd.DataFrame([df_lambda.to_dict(orient = 'index')])
    df_lambda = df_lambda.T.rename({0:'Keywords'}, axis = 1)
    
    if n is None:
        return df_lambda
    else:
        falpha = lambda alpha: sorted(alpha.items(), key=lambda x:x[1], reverse=True)[:n]
        df_lambda['Keywords'] = df_lambda['Keywords'].apply(falpha)
        return df_lambda

'''for training and storing models and vectorizers'''
def model_score_df_all(model_dict, category, folder_label, X_train, X_test, y_train, y_test):   
    

    models, model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], [], []
    
    for k,v in model_dict.items():   

        
        v.fit(X_train, y_train)
        
        model_name.append(k)
        models.append(v)
        
        y_pred = v.predict(X_test)
#         ac_score_list.append(accuracy_score(y_test, y_pred))
#         p_score_list.append(precision_score(y_test, y_pred, average='macro'))
#         r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
#         model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
#         model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
#         model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
    results = dict(zip(models, f1_score_list))
    name = dict(zip(model_name, f1_score_list))    
    #return best performing model according to f1_score
    best_clf = max(results.items(), key=operator.itemgetter(1))[0]
    best_f1 = max(results.items(), key=operator.itemgetter(1))[1]
    best_name = max(name.items(), key=operator.itemgetter(1))[0]
    
    print("best classifier model:", best_name)
    print("f1_score:", best_f1)

    #save best performing model
    filename = '../models/tf_idf/'+folder_label+'/'+category+'_'+best_name+'_'+'model.sav'
    joblib.dump(best_clf, filename)

    #save best performing model without name appendix
    #gfilename = '../models/tf_idf/'+folder_label+'/'+category+'_'+'model.sav'
    joblib.dump(best_clf, filename)      
    
    return results, best_f1
