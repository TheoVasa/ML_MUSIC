import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    #----------------------------------------------------------------------------------------
    ################################## PRINCIPAL METHODS ####################################
    #----------------------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.task_kind='regression'
        self.set_arguments(*args,**kwargs)
    #----------------------------------------------------------------------------------------
    def set_arguments(self,*args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            In case of ridge regression, you need to define lambda regularizer(lmda).

            You can either pass these as args or kwargs.
        """
        if "lmda" in kwargs:
            self.lmda = kwargs["lmda"]
        elif len(args) >0 :
            self.lmda = args[0]
        else:
            self.lmda = 0
    #----------------------------------------------------------------------------------------
    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_regression_targets (np.array): predicted target of shape (N,regression_target_size)
        """

        inverse = np.linalg.inv(np.transpose(training_data)@training_data + (self.lmda*np.identity(training_data.shape[1])))
        self.w = inverse@(np.transpose(training_data)@training_labels)
        pred_regression_targets = np.dot(training_data,self.w)
        return pred_regression_targets
    #----------------------------------------------------------------------------------------    
    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                pred_regression_targets (np.array): predicted targets of shape (N,regression_target_size)
        """   
        pred_regression_targets=np.dot(test_data,self.w)
        return pred_regression_targets
