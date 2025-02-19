from src.model import MultimodalClassifier
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm.auto import trange
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from dotenv import load_dotenv
from torchinfo import summary
from datetime import datetime
from collections import deque
import shutil


class EarlyStopping:
    def __init__(self, patience, epsilon):
        self.patience = patience
        self.epsilon = epsilon
        self.early_stop = False
        self.queue_train_losses = deque(maxlen=patience)
        self.queue_val_losses = deque(maxlen=patience)

    def __call__(self, current_train_loss, current_val_loss):
        self.queue_train_losses.append(current_train_loss)
        self.queue_val_losses.append(current_val_loss)
        if len(self.queue_train_losses) < self.patience or len(self.queue_val_losses) < self.patience:
            return

        # Early stop due to negligible training loss changes OR due to increasing validation loss
        consecutive_diffs_below_epsilon = abs(np.diff(self.queue_train_losses)) < self.epsilon
        consecutive_diffs_positive = np.diff(self.queue_val_losses) >= 0
        self.early_stop = np.all(consecutive_diffs_below_epsilon) or np.all(consecutive_diffs_positive)


class TrainEvalModel:
    def __init__(
            self,
            dataset,
            audio_vectorizer,
            text_vectorizer,
            lstm_n_layers=1,
            lstm_hidden_dim=256,
            fc_hidden_dim=128,
            n_epochs=50,
            lr=0.001,
            reset_log_file=True,
            reset_tensorboard=True,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
    ):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.FILE_NAME = f'{dataset}_{audio_vectorizer}_{text_vectorizer}_{lstm_hidden_dim}_{fc_hidden_dim}'

        LOG_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'logs')
        os.makedirs(LOG_PATH, exist_ok=True)
        self.LOG_FILE_PATH = f'{LOG_PATH}/{self.FILE_NAME}.txt'
        if reset_log_file: open(self.LOG_FILE_PATH, 'w').close()

        WRITER_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'runs')
        os.makedirs(WRITER_PATH, exist_ok=True)
        self.WRITER_FILE_PATH = f'{WRITER_PATH}/{self.FILE_NAME}'
        self.reset_tensorboard = reset_tensorboard

        self.dataset = dataset
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.n_epochs = n_epochs
        self.audio_vectorizer = audio_vectorizer
        self.text_vectorizer = text_vectorizer
        self.lstm_n_layers = lstm_n_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.lr = lr
        self.device = torch.device(device)
        self.run_pipeline()

    def log(self, text, print_to_console=True):
        if print_to_console: print(text)
        with open(self.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    def tensorboard_add_model_graph(self):
        with torch.no_grad():
            x_audio, x_text, _ = next(iter(self.train_dataset))
            x_audio = x_audio.to(self.device)
            x_text = x_text.to(self.device)
            self.writer.add_graph(self.model, input_to_model=(x_audio, x_text))

    def get_progressbar_description(self, phase, epoch, idx, n_samples, loss):
        desc = f'{phase}: Epoch [{epoch + 1}/{self.n_epochs}], ' \
               f'Item [{idx + 1}/{n_samples}], ' \
               f'Loss {loss.item():.3f}'
        return desc

    def compute_gradient_norm(self):
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def compute_metrics(self, labels, predictions, phase, epoch=None, n_decimals=3):
        accuracy = round(accuracy_score(labels, predictions), n_decimals)
        precision = round(precision_score(labels, predictions, zero_division=0), n_decimals)
        recall = round(recall_score(labels, predictions, zero_division=0), n_decimals)
        f1 = round(f1_score(labels, predictions, zero_division=0), n_decimals)

        metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        self.log(f'{phase} Metrics: {metrics}')
        if phase != 'Test':
            self.writer.add_scalars(main_tag=f'{phase} Metrics', tag_scalar_dict=metrics, global_step=epoch)
        return accuracy, precision, recall, f1

    def run_pipeline(self):
        start_time = datetime.now()
        vectorizers = {'audio_vectorizer': self.audio_vectorizer, 'text_vectorizer': self.text_vectorizer}

        if self.dataset == 'DAIC_WoZ':
            torch.backends.cudnn.enabled = False
            if self.reset_tensorboard and os.path.exists(self.WRITER_FILE_PATH):
                shutil.rmtree(self.WRITER_FILE_PATH)
            self.writer = SummaryWriter(self.WRITER_FILE_PATH)

            self.train_dataset = DAICWoZDataset(train_val_test='train', **vectorizers)
            self.val_dataset = DAICWoZDataset(train_val_test='val', **vectorizers)
            self.test_dataset = DAICWoZDataset(train_val_test='test', **vectorizers)
            self.train_and_evaluate()
            self.writer.close()

        elif self.dataset == 'Androids_Corpus':
            fold_metrics = list()
            for fold in range(5):
                writer_path = f'{self.WRITER_FILE_PATH}_fold_{fold}'
                if self.reset_tensorboard and os.path.exists(writer_path):
                    shutil.rmtree(writer_path)
                self.writer = SummaryWriter(writer_path)

                self.train_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='train', **vectorizers)
                self.val_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='val', **vectorizers)
                self.test_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='test', **vectorizers)
                accuracy, precision, recall, f1 = self.train_and_evaluate()
                fold_metrics.append([accuracy, precision, recall, f1])
                self.writer.close()

            accuracy, precision, recall, f1 = np.mean(fold_metrics, axis=0)
            self.log(f"{'=' * 14} 5-fold Cross Validation Test Results {'=' * 13}")
            self.log(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

        end_time = datetime.now()
        self.log(f'Pipeline Execution Time: {str(end_time - start_time).split('.')[0]}')

    def train_and_evaluate(self):
        # Model initialization
        self.model = MultimodalClassifier(
            audio_feature_dim=self.train_dataset.audio_feature_dim,
            text_feature_dim=self.train_dataset.text_feature_dim,
            lstm_n_layers=self.lstm_n_layers,
            lstm_hidden_dim=self.lstm_hidden_dim,
            fc_hidden_dim=self.fc_hidden_dim
        ).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)
        self.stopper = EarlyStopping(patience=4, epsilon=1e-2)
        self.log(str(summary(self.model)), print_to_console=False)
        self.tensorboard_add_model_graph()

        # ------------------------------------------- Training -------------------------------------------
        progressbar = trange(self.n_epochs)
        for epoch in progressbar:
            # Training Phase
            train_labels, train_predictions, train_loss = self.run_epoch(progressbar, phase='Training', epoch=epoch)

            # Validation Phase
            with torch.no_grad():
                val_labels, val_predictions, val_loss = self.run_epoch(progressbar, phase='Validation', epoch=epoch)

            # Compute the metrics, create the logs, and add their plots to tensorboard
            lr = self.scheduler.get_last_lr()[0]
            losses = {'Train': train_loss, 'Validation': val_loss}
            self.log(f"Epoch [{epoch + 1}/{self.n_epochs}]:")
            self.log(f"Learning Rate: {lr:.7f}")
            self.log(f"Training Loss: {losses['Train']:.3f}, Validation Loss: {losses['Validation']:.3f}")
            self.compute_metrics(train_labels, train_predictions, phase='Training', epoch=epoch)
            self.compute_metrics(val_labels, val_predictions, phase='Validation', epoch=epoch)
            self.writer.add_scalar(tag='Learning Rate', scalar_value=lr, global_step=epoch)
            self.writer.add_scalars(main_tag='Loss', tag_scalar_dict=losses, global_step=epoch)
            self.log('-' * 65)

            # Change the learning rate according to the scheduler, check conditions for early stopping
            self.scheduler.step()
            self.stopper(train_loss, val_loss)
            if self.stopper.early_stop:
                self.log('Early stopping was triggered!')
                break

        # ------------------------------------------ Evaluation ------------------------------------------
        # Test Phase
        with torch.no_grad():
            test_labels, test_predictions, _ = self.run_epoch(progressbar, phase='Test')

        # Compute the test metrics, add to log, no plotting is needed in tensorboard
        self.log(f"{'=' * 26} Test Results {'=' * 25}")
        return self.compute_metrics(test_labels, test_predictions, phase='Test')

    def run_epoch(self, progressbar, phase, epoch=None):
        if phase == 'Training':
            self.model.train()
            dataset = self.train_dataset
            backward_pass = True
            update_progress_bar = True
        elif phase == 'Validation':
            self.model.eval()
            dataset = self.val_dataset
            backward_pass = False
            update_progress_bar = True
        elif phase == 'Test':
            self.model.eval()
            dataset = self.test_dataset
            backward_pass = False
            update_progress_bar = False
        else:
            raise ValueError('Invalid phase')

        labels = []
        predictions = []
        total_loss = 0.0
        gradient_norm = 0.0

        for idx, (x_audio, x_text, y) in enumerate(dataset):
            x_audio, x_text, y = x_audio.to(self.device), x_text.to(self.device), y.to(self.device)

            # Forward pass and collect predictions, empty gradients for training
            if backward_pass: self.optimizer.zero_grad()
            output = self.model(x_audio, x_text).squeeze(1)
            pred = torch.sigmoid(output).round()
            predictions.append(pred.item())
            labels.append(y.item())

            # Calculate and accumulate loss
            loss = self.criterion(output, y)
            total_loss += loss.item()
            if update_progress_bar:
                n_samples = len(dataset)
                progressbar.set_description(self.get_progressbar_description(phase, epoch, idx, n_samples, loss))

            # Backward pass only for training
            if backward_pass:
                loss.backward()
                gradient_norm += self.compute_gradient_norm()
                self.optimizer.step()

        if backward_pass:
            self.writer.add_scalar(tag='Gradient Norm', scalar_value=gradient_norm, global_step=epoch)
        return labels, predictions, total_loss
