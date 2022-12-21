import numpy as np

class PCA(object):

    """
        PCA dimensionality reduction object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, find_principal_components, and reduce_dimension work correctly.
    """
    def __init__(self, *args, **kwargs):
        """
            You don't need to initialize the task kind for PCA.
            Call set_arguments function of this class.
        """
        self.set_arguments(*args, **kwargs)
        #the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        #the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The PCA class should have a variable defining the number of dimensions (d).
            You can either pass this as an arg or a kwarg.
        """
        if "D" in kwargs:
            self.D = kwargs["D"]
        elif len(args) >0 :
            self.D = args[0]
        else:
            self.D = 231

        if "max_exp_var" in kwargs:
            self.max_exp_var = kwargs["max_exp_var"]
        elif len(args) >0 :
            self.max_exp_var = args[0]
        else:
            self.max_exp_var = 0.9


    def find_principal_components(self, training_data):
        """
            Finds the principal components of the training data. Returns the explained variance in percentage.
            IMPORTANT: 
            This function should save the mean of the training data and the principal components as
            self.mean and self.W, respectively.

            Arguments:
                training_data (np.array): training data of shape (N,D)
            Returns:
                exvar (float): explained variance
        """
        # Compute the mean of data
        #This vector is a vector of dimension D (mean of the row vectors of X)
        self.mean = np.mean(training_data,axis=0)
        # Center the data with the mean
        #X of size NxD and mean of size Dx1, we use broadcasting to substract to each row the value of the mean 
        X_tilde = training_data-self.mean
        # Create the covariance matrix
        # N is the number of samples
        N = training_data.shape[0]
        C = (1/N)*(X_tilde.T@X_tilde)
        # Compute the eigenvectors and eigenvalues. Hint: use np.linalg.eigh
        eigvals, eigvecs = np.linalg.eigh(C)
        # Choose the top d eigenvalues and corresponding eigenvectors. Sort the eigenvalues( with corresponding eigenvectors )
        # in decreasing order first.
        eigvals = np.flip(eigvals)
        eigvecs = np.flip(eigvecs,axis=1)

        self.d = compute_dimension(eigvals, self.max_exp_var, self.D)

        # Create matrix W and the corresponding eigen values
        self.W = eigvecs[:,:self.d]
        eg = eigvals[:self.d]

        # Compute the explained variance
        exvar = compute_exvar(eigvals, self.d)

        return exvar*100

    def reduce_dimension(self, data):
        """
            Reduce the dimensions of the data, using the previously computed
            self.mean and self.W. 

            Arguments:
                data (np.array): data of shape (N,D)
            Returns:
                data_reduced (float): reduced data of shape (N,d)
        """
        Xtilde=data-self.mean
        return Xtilde@self.W

def compute_exvar(eigvals, d) :
    return  np.sum(eigvals[:d])/np.sum(eigvals)

def compute_dimension(eigvals, max_exp_var, big_d) :
    print("Searching best dimension reduction d s.t explained variance is bigger that", max_exp_var)
    d = big_d
    exvar = compute_exvar(eigvals, d)
    while(d > 1 and exvar > max_exp_var):
        d = d - 1 
        exvar = compute_exvar(eigvals, d)
    d = d + 1 
    exvar = compute_exvar(eigvals, d)
    print("The best dimension reduction is d =", d, " with an explained variance of", exvar)
    return d 



  
        

