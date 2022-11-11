import numpy as np 
import argparse

# these will be imported in MS2. uncomment then!
#import torch
#from torch.utils.data import DataLoader
#from methods.deep_network import SimpleNetwork, Trainer

from data import H36M_Dataset, FMA_Dataset, Movie_Dataset
from methods.pca import PCA
from methods.cross_validation import cross_validation
from metrics import accuracy_fn,mse_fn, macrof1_fn
from methods.knn import KNN
from methods.dummy_methods import DummyClassifier, DummyRegressor
from methods.logistic_regression import LogisticRegression
from methods.linear_regression import LinearRegression

#----------------------------------------------------------------------------------------
############################### EXTERNAL METHODS ########################################
#----------------------------------------------------------------------------------------
def append_bias_term(X_train):
    ones_column = np.ones((X_train.shape[0], 1))
    X_train_bias = np.concatenate([X_train, ones_column], axis=1)
    return X_train_bias
#----------------------------------------------------------------------------------------
#########################################################################################
#----------------------------------------------------------------------------------------

def main(args):
    # First we create all of our dataset objects. The dataset objects store the data, labels (for classification) and the targets for regression
    if args.dataset=="h36m":
        train_dataset = H36M_Dataset(split="train", path_to_data=args.path_to_data)
        test_dataset = H36M_Dataset(split="test", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)
        #uncomment for MS2
        #val_dataset = H36M_Dataset(split="val",path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)

    elif args.dataset=="music":
        train_dataset = FMA_Dataset(split="train", path_to_data=args.path_to_data)
        test_dataset = FMA_Dataset(split="test", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)
        #uncomment for MS2
        #val_dataset = FMA_Dataset(split="val",path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)
        
    elif args.dataset=="movies":
        train_dataset = Movie_Dataset(split="train", path_to_data=args.path_to_data)
        test_dataset = Movie_Dataset(split="test", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)
        #uncomment for MS2
        #val_dataset = Movie_Dataset(split="val", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)

    # Note: We only use the following methods for more old-school methods, not the nn!
    train_data, train_regression_target, train_labels = train_dataset.data, train_dataset.regression_target, train_dataset.labels
    test_data, test_regression_target, test_labels = test_dataset.data, test_dataset.regression_target, test_dataset.labels

    print("Dataloading is complete!")

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d = 200)
        pca_obj.find_principal_components(train_data)
        train_data = pca_obj.reduce_dimension(train_data)
        train_regression_target = pca_obj.reduce_dimension(train_regression_target)
        test_data = pca_obj.reduce_dimension(test_data)
        test_regression_target = pca_obj.reduce_dimension(test_regression_target)

    # Neural network. (This part is only relevant for MS2.)
    if args.method_name == "nn":
        # Pytorch dataloaders
        print("Using deep network")
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # create model
        model = SimpleNetwork(input_size=train_dataset.feature_dim, num_classes=train_dataset.num_classes, regression_output_size=train_dataset.regression_target_size)
        
        # training loop
        trainer = Trainer(model, lr=args.lr, epochs=args.max_iters)
        trainer.train_all(train_dataloader, val_dataloader)
        results_class, results_reg = trainer.eval(test_dataloader)
        torch.save(results_class, "results_class.txt")
        torch.save(results_reg, "results_reg.txt")
    
    # classical ML methods (MS1 and MS2)
    # we first create the classification/regression objects
    # search_arg_vals and search_arg_name are defined for cross validation
    # we show how to create the objects for DummyClassifier and DummyRegressor
    # the rest of the methods are up to you!
    else:
        if args.method_name == "logistic_regression":
            if args.use_cross_validation:
                #don't enter specific hyperparameters 
                method_obj =  LogisticRegression(lr=args.lr, max_iters=args.max_iters, nbr_classes=3)
                #use with cross validation              
                #search_arg_vals = np.arange(args.start_range, args.end_range, args.step_range)
                search_arg_vals=np.logspace(-7,-4,4)
                #change depending of which hyperparameter perform cross validation
                #search_arg_name = "max_iters"
                search_arg_name = "lr"     
            else :
                #enter the parameters 
                method_obj =  LogisticRegression(lr=args.lr, max_iters=args.max_iters, nbr_classes=3)

            #the output is classification 
            output_training_target = train_labels

        elif args.method_name == 'linear_regression':
            method_obj = LinearRegression()
            #append bias term (not mandatory, they didn't used it)
            """"
            train_data = append_bias_term(train_data)
            test_data = append_bias_term(test_data)
            """
      
            #the output is regression (rating)
            output_training_target = train_regression_target   

        elif args.method_name == 'ridge_regression':  
            if args.use_cross_validation:
                #don't enter specific hyperparameters 
                method_obj = LinearRegression()
                #use with cross validation      
                #search_arg_vals = np.arange(args.start_range, args.end_range, args.step_range)
                #search_arg_vals=np.logspace(-5,1,7)
                search_arg_vals=np.arange(0,20)
                search_arg_name = "lmda"     
            else : 
                method_obj = LinearRegression(lmda=args.ridge_regression_lmda)

            #append bias term (not mandatory, they didn't used it)
            train_data = append_bias_term(train_data)
            test_data = append_bias_term(test_data)

            #the output is regression (rating)
            output_training_target = train_regression_target   

        # cross validation (MS1)
        if args.use_cross_validation:
            print("Using cross validation")
            best_arg, best_val_acc = cross_validation(method_obj=method_obj, search_arg_name=search_arg_name, search_arg_vals=search_arg_vals, data=train_data, labels=train_labels, k_fold=4)
            print("Best value for " + str(search_arg_name) + " is " + str(best_arg) + " with an accuracy of " + str(best_val_acc))
            # set the classifier/regression object to have the best hyperparameter found via cross validation:
            arg_dict = {search_arg_name: best_arg}
            method_obj.set_arguments(**arg_dict)

        # FIT AND PREDICT:
        method_obj.fit(train_data, output_training_target)
        pred_labels = method_obj.predict(test_data)
        # Report test results
        if method_obj.task_kind == 'regression':
            loss = mse_fn(pred_labels,test_regression_target)
            print("Final loss is", loss)
        else:
            acc = accuracy_fn(pred_labels,test_labels)
            print("Final classification accuracy is", acc)
            macrof1 = macrof1_fn(pred_labels,test_labels)
            print("Final macro F1 score is", macrof1)
            
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="h36m", type=str, help="choose between h36m, movies, music")
    parser.add_argument('--path_to_data', default="..", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ..")
    parser.add_argument('--method_name', default="knn", type=str, help="knn / logistic_regression / nn")
    parser.add_argument('--knn_neighbours', default=3, type=int, help="number of knn neighbours")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--ridge_regression_lmda', type=float, default=1, help="lambda for ridge regression")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--use_cross_validation', action="store_true", help="to enable cross validation")
    parser.add_argument('--start_range', default=0, type=float, help="the starting of the range for hyper parameter of cross-validation")
    parser.add_argument('--end_range', default=1, type=float, help="the ending of the range for hyper parameter of cross-validation")
    parser.add_argument('--step_range', default=1, type=float, help="the step for the range of hyperparameter in cross-validation")
    

    # Feel free to add more arguments here if you need

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    args = parser.parse_args()
    main(args)


