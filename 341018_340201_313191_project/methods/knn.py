import numpy as np

class KNN(object):
    """
        kNN classifier object.
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

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The KNN class should have a variable defining the number of neighbours (k).
            You can either pass this as an arg or a kwarg.
        """
        if "k" in kwargs:
            self.k = kwargs["k"]
        elif len(args) >0 :
            self.k = args[0]
        else:
            self.k = 6 


    

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.data_train=training_data
        self.labels_train=training_labels

        pred_labels=self.kNN(self.data_train,self.data_train,self.labels_train,self.k)
        return pred_labels
                               
    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """      
        test_labels=self.kNN(test_data,self.data_train,self.labels_train,self.k)
        return test_labels

    #----------------------------------------------------------------------------------------
    def euclidean_dist(self,example, training_examples):
        """function to compute the Euclidean distance between a single example
        vector and all training_examples

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            return distance vector of length N
        """
        return np.linalg.norm(example-training_examples,axis=1)
    #----------------------------------------------------------------------------------------
    def find_k_nearest_neighbors(self,k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
        Tip: use np.argsort()
        """
        indices = np.argsort(distances)[:k]
    
        return indices
    #----------------------------------------------------------------------------------------
    def predict_label(self,neighbor_labels):
        """return the most frequent element in the input.
        """
        return np.argmax(np.bincount(neighbor_labels))
    #----------------------------------------------------------------------------------------
    def kNN_one_example(self,unlabeled_example, training_features, training_labels, k):
        """returns the label of single unlabelled_example.
        """
    
        # Compute distances
        distances = self.euclidean_dist(unlabeled_example,training_features)
    
        # Find neighbors
        nn_indices = self.find_k_nearest_neighbors(k,distances)
    
        # Get neighbors' labels
        neighbor_labels = np.take(training_labels,nn_indices)
    
        # Pick the most common
        best_label = self.predict_label(neighbor_labels)
    
        return best_label
    def kNN(self,unlabeled, training_features, training_labels, k):
        """return the labels vector for all unlabeled datapoints.
        """
        return np.apply_along_axis(func1d=self.kNN_one_example,axis=1,arr=unlabeled,training_features=training_features,training_labels=training_labels,k=k)    
   

