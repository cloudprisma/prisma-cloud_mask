from sklearn.neighbors import KNeighborsClassifier 
import sys


def knn_classf(X_train, y_train):
	
    """
    Goals: setup parameters for K-Nearest Neighbor Classifier.
           
    Parameters
    ----------
        X_train, y_train: full dataset   
    Returns:   
    ----------
        knn_clf: trained model with selected parameters
    
    """ 
    params = {}
    params['n_neighbors'] = 5
    params["weights"] = "distance"     
    params['algorithm'] = 'ball_tree'       	
    # Instantiate K Nearest Neighbors
    knn_clf = KNeighborsClassifier(n_jobs=-1)
    knn_clf.set_params(**params)

    # Fit on training data
    knn_clf.fit(X_train, y_train)

    return knn_clf


if __name__ == '__main__':
    '''
    Main driver.
    '''
    sys.exit()
