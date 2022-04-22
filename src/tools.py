from typing import Iterable, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.utils import class_weight

from collections import Counter

from sklearn.metrics import accuracy_score, confusion_matrix

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
                         columns = vectorizer.get_feature_names(),
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