from sklearn.metrics import balanced_accuracy_score, accuracy_score, auc
import sys
import xgboost as xgb


def xgboost_classf(X_train, y_train):

    """
    Goals: setup parameters for xgboost class using a dictionary format.
           
    Parameters
    ----------
        X_train, y_train: full dataset
    
    Returns:   
    ----------
        xgb_clf: trained model with selected parameters
        train_accuracy: accuracy score on train data
    
    """     
    
    # Define booster parameters using a dictionary - basic parameters:
    # setup parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['booster'] = 'gbtree'
    # params["device"] = "cuda"     
    params['n_estimators'] = 56     
    
    # Tree complexity parameters
    params["tree_method"] = 'approx'
    params['max_depth'] = 11
    params['min_child_weight'] = 1
    # Sampling parameters
    params['learning_rate'] = 0.3  
    params["subsample"] =  0.873313097133705
    params['colsample_bytree'] = 0.568629233715412
    params['reg_lambda'] = 0.8032300475995502
    xgb_clf = xgb.XGBClassifier(n_jobs=-1)
    xgb_clf.set_params(**params)
    
    # Fit on training data   
    xgb_clf.fit(X_train, y_train)

    return xgb_clf

    
    
    
if __name__ == '__main__':
    '''
    Main driver.
    '''
    sys.exit()
