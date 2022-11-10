import numpy as np
from metrics import accuracy_fn, mse_fn, macrof1_fn

def splitting_fn(data, labels, indices, fold_size, fold):
    """
        Function to split the data into training and validation folds.
        Arguments:
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            indices: (np.array, of shape (N,)): array of pre shuffled indices (integers ranging from 0 to N)
            fold_size (int): the size of each fold
            fold (int): the index of the current fold.
        Returns:
            train_data, train_label, val_data, val_label (np. arrays): split training and validation sets
    """
    index=fold*fold_size
    mask=np.ones(data.shape[0],dtype=bool)
    mask[index:(index+fold_size)]=False
    train_indices=indices[mask]
    val_indices=indices[~mask]

    mask=np.ones(data.shape[0],dtype=bool)
    mask[val_indices]=False
    train_data=data[mask]
    train_label=labels[mask]
    val_data=data[~mask]
    val_label=labels[~mask]
    
    return train_data, train_label, val_data, val_label

def cross_validation(method_obj=None, search_arg_name=None, search_arg_vals=[], data=None, labels=None, k_fold=4):
    """
        Function to run cross validation on a specified method, across specified arguments.
        Arguments:
            method_obj (object): A classifier or regressor object, such as KNN. Needs to have
                the functions: set_arguments, fit, predict.
            search_arg_name (str): the argument we are trying to find the optimal value for
                for example, for DummyClassifier, this is "dummy_arg".
            search_arg_vals (list): the different argument values to try, in a list.
                example: for the "DummyClassifier", the search_arg_name is "dummy_arg"
                and the values we try could be [1,2,3]
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            k_fold (int): number of folds
        Returns:
            best_hyperparam (float): best hyper-parameter value, as found by cross-validation
            best_acc (float): best metric, reached using best_hyperparam
    """
    ## choose the metric and operation to find best params based on the metric depending upon the
    ## kind of task.
    metric = mse_fn if method_obj.task_kind == 'regression' else macrof1_fn
    find_param_ops = np.argmin if method_obj.task_kind == 'regression' else np.argmax
    #preparing splitting of the data 
    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_size = N//k_fold

    #setting the list of accuracy for each hyperparameter(s)
    acc_by_param = []

    for arg in search_arg_vals:
        #set argument(s)
        arg_dict = {search_arg_name: arg}
        method_obj.set_arguments(**arg_dict)
        #setting the list of accuracy for each folds 
        acc_by_folds = []
        for fold in range(k_fold):
            #split into train and validation data depending of the current fold 
            train_data,train_label,val_data,val_label = splitting_fn(data,labels,indices,fold_size,fold)
            #train the data
            method_obj.fit(train_data,train_label)
            #test the accuracy of this prediction 
            val_pred = method_obj.predict(val_data)
            measure=metric(val_pred,val_label)
            acc_by_folds.append(measure)
        
        acc_by_param.append(np.mean(acc_by_folds))
        
    #deducting the best hyperparameter(s)
    best_index = find_param_ops(acc_by_param)
    best_hyperparam = search_arg_vals[best_index]
    best_acc=acc_by_param[best_index]    
         
    return best_hyperparam, best_acc

        


    