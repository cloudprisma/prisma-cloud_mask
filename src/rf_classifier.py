from sklearn.ensemble import RandomForestClassifier
import sys


def rf_classf(X_train, y_train):
	
    """
    Goals: setup parameters for RandomForest Classifier
           
    Parameters
    ----------
        X_train, y_train: full dataset
        n_estimators: number of decision trees to use
    Returns:   
    ----------
        rf_clf: trained model with selected parameters    
    """ 
    
    
    params = {}
    params['criterion'] = 'gini'
    params["n_estimators"] = 43     
    params['max_depth'] = 24    
    params["max_features"] = 3
    params['min_samples_split'] = 4  
    # Instantiate classifier
    rf_clf = RandomForestClassifier(n_jobs=-1)
    rf_clf.set_params(**params)
   
    # Fit on training data
    rf_clf.fit(X_train, y_train)

    return rf_clf


if __name__ == '__main__':
    '''
    Main driver.
    '''
    sys.exit()
