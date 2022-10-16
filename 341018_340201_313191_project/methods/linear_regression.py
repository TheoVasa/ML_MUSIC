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

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.task_kind='regression'
        self.set_arguments(*args,**kwargs)
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

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
            self.lmda = 1


        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
    

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_regression_targets (np.array): predicted target of shape (N,regression_target_size)
        """
        training_data2 = self.append_bias_term(training_data)
        self.w = np.linalg.pinv(training_data2) @ training_labels
        pred_regression_targets = np.dot(training_data2,self.w)
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return pred_regression_targets

    def append_bias_term(X_train):
        N=X_train.shape[0]
        ones_column = np.ones((N,1))
        X_train_bias = np.concatenate((ones_column,X_train),axis=1)
        return X_train_bias    

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                pred_regression_targets (np.array): predicted targets of shape (N,regression_target_size)
        """   

        test_data2=self.append_bias_term(test_data)
        pred_regression_targets=np.dot(test_data2,self.w)

        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return pred_regression_targets
