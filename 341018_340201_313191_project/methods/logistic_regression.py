import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot


class LogisticRegression(object):
    """
        LogisticRegression classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.task_kind='classification'
        self.set_arguments(*args,**kwargs)
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """
        
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        elif len(args) >0 :
            self.lr = args[0]
        else:
            self.lr = 1

        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        elif len(args) >0 :
            self.max_iters = args[1]
        else:
            self.max_iters = 1
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
       
    def softmax(data,w):
        res=np.exp(data@w)/(np.sum(np.exp(data@w),axis=1).reshape((-1, 1)))
        return res

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        training_data2 = self.append_bias_term(training_data)
        self.w = np.linalg.pinv(training_data2) @ training_labels

        pred_labels=self.classify(training_data)



        
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return pred_labels

    def append_bias_term(X_train):
        N=X_train.shape[0]
        ones_column = np.ones((N,1))
        X_train_bias = np.concatenate((ones_column,X_train),axis=1)
        return X_train_bias

    def classify(self,data):
        proba=self.softmax(data,self.w)
        pred_labels=np.zeros(proba.shape)
        pred_labels[np.argmax(proba)]=1
        return pred_labels     


    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   

        test_data2=self.append_bias_term(test_data)
        
        pred_labels=self.classify(test_data)


        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return pred_labels
