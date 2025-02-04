from torch.utils.data import DataLoader


class TrainModel:
    def __init__(self, dataset, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
        self.dataset = dataset
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
