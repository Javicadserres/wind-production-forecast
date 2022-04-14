import torch


class Trainer(object):
    def __init__(self, model, criterion, optimizer):
        """
        Class that facilitates the training of a pytorch model.
        
        Parameters
        ----------
        model: torch.Module
        criterion: torch.Module
        optimizer: torch.optim
        
        Attributes
        ----------
        train_losses: list
        val_losses: list
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
    def fit(self, train_loader, val_loader, epochs, patience=100):
        """Fit the model.
        
        Parameters
        ----------
        train_loader: torch.DataLoader
        val_loader: torch.DataLoader
        epochs: int
        patience: int
        
        Returns
        -------
        model: torch.Module
        """ 
        self._initialize_variables(
                train_loader, val_loader, epochs, patience
        )
        while self.epoch < epochs and not self.stop: 
            # training
            train_loss = self.forward_train()                
            self.train_losses.append(train_loss)   
            
            # validation
            valid_loss = self.forward_validation()
            self.val_losses.append(valid_loss)
            
            # logger and earlystopping
            self.logger()
            self.earlystopping() 
            
            # update epoch
            self.epoch += 1
            
        return self.model

    def _initialize_variables(
            self, train_loader, val_loader, epochs, patience
        ):
        """
        Initialize the variables for the training processes.
        """
        self.last_loss = torch.inf
        self.trigger = 0
        self.train_losses = []
        self.val_losses = []
        self.stop = False
        self.epoch = 0
        self.patience = patience
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

    def forward_train(self):
        """
        Forward and backward pass in the train set.
        """
        self.model.train()
        train_loss = 0

        for num, data in enumerate(self.train_loader):
            inputs, target = data
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, target)

            # backward propagation and update
            loss.backward()
            self.optimizer.step()    
            train_loss += loss.item()

        return train_loss / num

    def forward_validation(self):
        """
        Forward pass in the validation set.
        """
        self.model.train(False)
        valid_loss = 0

        for num, data in enumerate(self.val_loader, 0):
            inputs, target = data

            # forward and loss
            outputs = self.model(inputs)
            loss = self.criterion(outputs, target)
            valid_loss += loss.item()

        return valid_loss / num

    def logger(self):
        """
        Prints information about the training processes.
        """
        epoch_info = f'epoch: {self.epoch} '
        train_info = f'train loss: {round(self.train_losses[-1], 3)} '
        val_info = f'val loss: {round(self.val_losses[-1], 3)}'

        print(epoch_info + train_info + val_info)

    def earlystopping(self):
        """
        Early stopping method.
        """
        # check actual loss
        if self.val_losses[-1] >= self.last_loss: self.trigger += 1
        self.last_loss = self.val_losses[-1]
        
        # condition for stopping
        if self.trigger >= self.patience: self.stop = True
