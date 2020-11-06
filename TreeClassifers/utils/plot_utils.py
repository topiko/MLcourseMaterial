import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def plot_decision_surfaces(clf, X_train, y_train, X_test, y_test, ax=None):
    """
    Plot the decision surfaces of given classifier when only
    2-features are present.
    Args:
        clf: classifier for which the surfaces are plotted:
        X_train: training data shape (N, 2).
        y_train: training data labels.
        X_test: test data shape (M, 2).
        y_test: test data labels.
        ax: axes to which plot, if None - create figure
    Return:
        ax: the axes with decision surfaces plotted.
    """
    
    
    if ax is None:
        _, ax = plt.subplots()
    
    if isinstance(X_train, pd.DataFrame):
        xlabel, ylabel = X_train.columns
        X_train = X_train.values
        X_test = X_test.values
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    
    clf.fit(X_train, y_train)
    
    # Create a meshgrids of x and y values
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    Z = clf.predict(np.vstack((xx.ravel(), yy.ravel())).T)
    Z = Z.reshape(xx.shape)
    
    cs = ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    
    # Plot the training points
    for i, color in enumerate('ryb'):
        mask = y_train == i
        ax.scatter(X_train[mask, 0], X_train[mask, 1], 
                    c=color, label=le.inverse_transform([i])[0],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Extract the clf score in the test set: (remember to transform the test set labels since the classifier was trained with tranformed rtain set ealier.
    metric_value = accuracy_score(le.transform(y_test), clf.predict(X_test))
    ax.set_title('Test set {}={:.3f}'.format('accuracy', metric_value))
    ax.legend()
    
    return ax

def decision_surfaces_row(clf, X_train, y_train, X_test, y_test, feature_pair_list):
    fig, axrow = plt.subplots(1,len(feature_pair_list))
    for features, ax in zip(feature_pair_list, axrow):
        plot_decision_surfaces(clf, X_train.loc[:, features], y_train, X_test.loc[:, features], y_test, ax=ax)
    fig.suptitle('{} Decision Surfaces:'.format(clf))
