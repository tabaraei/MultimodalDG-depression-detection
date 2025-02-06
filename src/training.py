from torch.utils.data import DataLoader


class TrainModel:
    def __init__(self, train_dataset, test_dataset, model, criterion, optimizer, scheduler, n_epochs, batch_size):
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=train_dataset.collate_fn
        )
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs

    def train(self):
        for epoch in self.n_epochs:
