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
    #to not reset other args when calling set arguments again 
    max_iters_set = False 
    lr_set = False
    #----------------------------------------------------------------------------------------
    ################################## PRINCIPAL METHODS ####################################
    #----------------------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.task_kind='classification'
        self.set_arguments(*args,**kwargs)
    #----------------------------------------------------------------------------------------
    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """

        #setting the rate for logistic
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
            self.lr_set = True
        elif not self.lr_set:
            self.lr = 0.0001
            self.lr_set = True

        #setting the max of iterations for logistic 
        if "max_iters" in kwargs :
            self.max_iters = kwargs["max_iters"]
            self.max_iters_set = True
        elif not self.max_iters_set:
            self.max_iters = 20
            self.max_iters_set = True 
        
        #setting the number of classes for classification
        if "nbr_classes" in kwargs: 
            self.nbr_classes = kwargs["nbr_classes"]
        else: 
            self.nbr_classes = 3
    #----------------------------------------------------------------------------------------
    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """

        #training the data / optimizing the weights 
        self.w = self.logistic_regression_train_multi(training_data, training_labels, self.max_iters, self.lr)
        #prediction on the training data 
        return self.logistic_regression_classify_multi(training_data, self.w)
     #----------------------------------------------------------------------------------------
    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """    

        return self.logistic_regression_classify_multi(test_data, self.w)
    #----------------------------------------------------------------------------------------
    ################################### INTERNALS HELPERS METHODS ###########################
    #----------------------------------------------------------------------------------------  
    def f_softmax(self,data,w):
        """ Softmax function
        Args:
            data (np.array): Input data of shape (N, D)
            w (np.array): Weights of shape (D, C) where C is # of classes
            
        Returns:
            res (np.array): Probabilites of shape (N, C), where each value is in 
                range [0, 1] and each row sums to 1.
        """

        #weighted_mat = np.vectorize(np.exp)(data @ w)
        weighted_mat=np.exp(data@w) 
        sum_mat = np.sum(weighted_mat, axis=1)
        for i in range(weighted_mat.shape[0]):
            for j in range(weighted_mat.shape[1]):
                weighted_mat[i,j] = weighted_mat[i,j] / sum_mat[i]
        return weighted_mat
        
    #----------------------------------------------------------------------------------------
    def gradient_logistic_multi(self, data, labels, w):
        """ Logistic regression gradient function for binary classes
        
        Args:
            data (np.array): Dataset of shape (N, D).
            labels (np.array): Labels of shape (N, C).
            w (np.array): Weights of logistic regression model of shape (D, C)
        Returns:
            grad (np. array): Gradient array of shape (D, C)
        """

        return data.T @ (self.f_softmax(data, w) - labels)

    #----------------------------------------------------------------------------------------
    def logistic_regression_train_multi(self, data, labels, k=16, max_iters=10, lr=0.001):
        """ Classification function for multi class logistic regression. 
        
        Args:
            data (np.array): Dataset of shape (N, D).
            labels (np.array): Labels of shape (N, )
            max_iters (integer): Maximum number of iterations. Default:10
            lr (integer): The learning rate of  the gradient step. Default:0.001

        Returns:
            np. array: trained weights of shape of shape (D, C).
        """
        #putting labels in onehot
        one_hot_labels = np.zeros([labels.shape[0], self.nbr_classes])
        one_hot_labels[np.arange(labels.shape[0]), labels.astype(int)] = 1
        labels = one_hot_labels
        weights = np.random.normal(0, 0.1, [data.shape[1], self.nbr_classes])
        for it in range(int(self.max_iters)):
            #update the weights
            weights = weights - self.lr * self.gradient_logistic_multi(data, labels, weights)
            predictions = self.logistic_regression_classify_multi(data, weights)
            #stop if we get 100%
            if self.accuracy_fn(self.onehot_to_label(labels), predictions) == 1:
                break
        
        return weights
    #----------------------------------------------------------------------------------------
    def logistic_regression_classify_multi(self, data, w):
        """ Classification function for multi class logistic regression. 
        
        Args:
            data (np.array): Dataset of shape (N, D).
            w (np.array): Weights of logistic regression model of shape (D, C)
        Returns:
            predictions (np.array): Label assignments of data of shape (N, ) (NOT one-hot!).
        """

        # find predictions, argmax to find the correct label
        return self.onehot_to_label(self.f_softmax(data, w))
    #----------------------------------------------------------------------------------------
    def accuracy_fn(self, labels_gt, labels_pred):
        """ Computes accuracy.
        
        Args:
            labels_gt (np.array): GT labels of shape (N, ).
            labels_pred (np.array): Predicted labels of shape (N, ).
            
        Returns:
            acc (float): Accuracy, in range [0, 1].
        """

        np.sum(np.abs(labels_gt - labels_pred)==0)
        return np.sum(labels_gt == labels_pred) / labels_gt.shape[0]
    #----------------------------------------------------------------------------------------
    def onehot_to_label(self, onehot):
        return np.argmax(onehot, axis=1)
    #----------------------------------------------------------------------------------------
    #########################################################################################
