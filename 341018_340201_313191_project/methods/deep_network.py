import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metrics import accuracy_fn, macrof1_fn

## MS2!!


class SimpleNetwork(nn.Module):
    """
    A network which does classification!
    """
    def __init__(self, input_size, num_classes):
        super(SimpleNetwork, self).__init__()
        self.input_size=input_size
        self.num_classes=num_classes
        #model : 
        # to define my nbr of neurons I just took more or less the mean between the input and to nbr of classes 
        # we could have used a more complexe formula with  hyperparameters and run cross-validation on it to optimize the model 
        self.fc1 = nn.Linear(self.input_size, 121)
        self.fc2 = nn.Linear(121 , 65)
        self.fc3 = nn.Linear(65 , 30)
        self.fc4 = nn.Linear(30 ,self.num_classes)

    def forward(self, x):
        """
        Takes as input the data x and outputs the 
        classification outputs.
        Args: 
            x (torch.tensor): shape (N, D)
        Returns:
            output_class (torch.tensor): shape (N, C) (logits)
        """
        #print(x.shape)
        
        x            = F.relu(self.fc1(x))
        x            = F.relu(self.fc2(x))
        x            = F.relu(self.fc3(x))
        # no activation fonction for last layer
        output_class = self.fc4(x) 
        return output_class

class Trainer(object):

    """
        Trainer class for the deep network.cd Desk
    """
    #OK
    def __init__(self, model, lr, epochs, beta=100):
        """
        """
        self.lr = lr
        self.epochs = epochs
        self.model= model
        self.beta = beta

        self.classification_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    #OK
    def train_all(self, dataloader_train, dataloader_val):
        """
        Method to iterate over the epochs. In each epoch, it should call the functions
        "train_one_epoch" (using dataloader_train) and "eval" (using dataloader_val).
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train)
            pred = self.eval(dataloader_val).numpy()
            #val_labels = dataloader_val.dataset.labels 
            #print("EPOCH " + str(ep) + "/" + str(self.epochs) + " accuracy : " + str(accuracy_fn(pred, val_labels)))

            if (ep+1) % 2 == 0:
                print("Reduce Learning rate")
                for g in self.optimizer.param_groups:
                    g["lr"] = g["lr"]*0.8

    #OK
    def train_one_epoch(self, dataloader):
        """
        Method to train for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode!
        i.e. self.model.train()
        """
        
        self.model.train()
        for it, batch in enumerate(dataloader):
            x, _ , labels = batch
            logits=self.model.forward(x)
            loss=self.classification_criterion(logits,labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval(self, dataloader):
        """
            Method to evaluate model using the validation dataset OR the test dataset.
            Don't forget to set your model to eval mode!
            i.e. self.model.eval()

            Returns:
                Note: N is the amount of validation/test data. 
                We return one torch tensor which we will use to save our results (for the competition!)
                results_class (torch.tensor): classification results of shape (N,)
        """
        self.model.eval()
        with torch.no_grad():
            results_class=[]
            for it, batch in enumerate(dataloader):
                x,_,_=batch
                pred = self.model(x)
                pred = np.argmax(pred, axis=1)
                results_class.append(pred)
        return torch.cat(results_class)

