from src.model import MultimodalClassifier
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import gc


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
        consecutive_diffs_positive = np.diff(self.queue_val_losses) > 0
        self.early_stop = np.all(consecutive_diffs_below_epsilon) or np.all(consecutive_diffs_positive)


class TrainEvalModel:
    def __init__(
            self,
            dataset,
            modality,
            audio_vectorizer,
            text_vectorizer,
            audio_lstm_hidden_dim,
            text_lstm_hidden_dim,
            fc_hidden_dim,
            lr,
            weight_decay,
            scheduler_factor,
            scheduler_patience,
            stopper_patience,
            n_epochs,
            device,
            segment_duration=None,
            reset_log_file=True,
            reset_tensorboard=True
    ):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.FOLDER_PATH = (
            f'{dataset}{f'_{segment_duration}s' if segment_duration else ''}/'
            f'{audio_vectorizer}_{text_vectorizer}/'
            f'{modality}'
        )
        self.FILE_NAME = (
            f'{audio_lstm_hidden_dim}_{text_lstm_hidden_dim}_{fc_hidden_dim}_{lr}_{weight_decay}_'
            f'{scheduler_factor}_{scheduler_patience}_{stopper_patience}_{n_epochs}'
        )

        LOG_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'logs', self.FOLDER_PATH)
        os.makedirs(LOG_PATH, exist_ok=True)
        self.LOG_FILE_PATH = f'{LOG_PATH}/{self.FILE_NAME}.txt'
        if reset_log_file: open(self.LOG_FILE_PATH, 'w').close()

        WRITER_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'runs', self.FOLDER_PATH)
        os.makedirs(WRITER_PATH, exist_ok=True)
        self.WRITER_FILE_PATH = f'{WRITER_PATH}/{self.FILE_NAME}'
        self.reset_tensorboard = reset_tensorboard

        self.dataset = dataset
        self.modality = modality
        self.audio_vectorizer = audio_vectorizer
        self.text_vectorizer = text_vectorizer
        self.segment_duration = segment_duration
        self.audio_lstm_hidden_dim = audio_lstm_hidden_dim
        self.text_lstm_hidden_dim = text_lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.stopper_patience = stopper_patience
        self.n_epochs = n_epochs
        self.device = device

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.run_pipeline()

    def log(self, text, print_to_console=True):
        if print_to_console: print(text)
        with open(self.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    def initialize_writer(self, PATH):
        if self.reset_tensorboard and os.path.exists(PATH):
            shutil.rmtree(PATH)
        self.writer = SummaryWriter(PATH)

    def tensorboard_add_model_graph(self):
        with torch.no_grad():
            x_audio, x_text, _ = next(iter(self.train_dataset))
            x_audio, x_text = x_audio.to(self.device), x_text.to(self.device)
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

    def clean_GPU_cache(self):
        del self.train_dataset, self.val_dataset, self.test_dataset, self.model
        torch.cuda.empty_cache()
        gc.collect()

    def overcome_GPU_memory_constraints(self):
        if self.dataset == 'DAIC_WoZ' and self.audio_vectorizer == 'HuBERT' and self.text_vectorizer == 'XLMRoBERTa':
            self.device = torch.device('cpu')
        torch.backends.cudnn.enabled = False if self.text_vectorizer == 'XLMRoBERTa' else True

    def run_pipeline(self):
        self.overcome_GPU_memory_constraints()
        start_time = datetime.now()
        args = {
            'audio_vectorizer': self.audio_vectorizer,
            'text_vectorizer': self.text_vectorizer,
            'segment_duration': self.segment_duration,
            'device': self.device
        }

        if self.dataset == 'DAIC_WoZ':
            self.initialize_writer(PATH=self.WRITER_FILE_PATH)
            self.train_dataset = DAICWoZDataset(train_val_test='train', **args)
            self.val_dataset = DAICWoZDataset(train_val_test='val', **args)
            self.test_dataset = DAICWoZDataset(train_val_test='test', **args)
            self.train_and_evaluate()
            self.writer.close()
            self.clean_GPU_cache()

        elif self.dataset == 'Androids_Corpus':
            fold_metrics = list()
            for fold in range(5):
                self.initialize_writer(PATH=f'{self.WRITER_FILE_PATH}_fold_{fold}')
                self.train_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='train', **args)
                self.val_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='val', **args)
                self.test_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='test', **args)
                accuracy, precision, recall, f1 = self.train_and_evaluate()
                fold_metrics.append([accuracy, precision, recall, f1])
                self.writer.close()
                self.clean_GPU_cache()

            accuracy, precision, recall, f1 = np.mean(fold_metrics, axis=0)
            self.log(f"{'=' * 14} 5-fold Cross Validation Test Results {'=' * 13}")
            self.log(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

        end_time = datetime.now()
        self.log(f'Pipeline Execution Time: {str(end_time - start_time).split('.')[0]}')

    def train_and_evaluate(self):
        # Model initialization
        pos_weight = (len(self.train_dataset) - sum(self.train_dataset.y)) / sum(self.train_dataset.y)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        self.model = MultimodalClassifier(
            modality=self.modality,
            audio_feature_dim=self.train_dataset.audio_feature_dim,
            text_feature_dim=self.train_dataset.text_feature_dim,
            audio_lstm_hidden_dim=self.audio_lstm_hidden_dim,
            text_lstm_hidden_dim=self.text_lstm_hidden_dim,
            fc_hidden_dim=self.fc_hidden_dim
        ).to(self.device)
        self.log(str(summary(self.model)), print_to_console=False)
        self.tensorboard_add_model_graph()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.stopper = EarlyStopping(patience=self.stopper_patience, epsilon=1e-2)
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=self.scheduler_patience,
            factor=self.scheduler_factor
        )

        # ------------------------------------------- Training -------------------------------------------
        progressbar = trange(self.n_epochs)
        for epoch in progressbar:
            # Training Phase
            train_labels, train_predictions, train_loss = self.run_epoch(progressbar, phase='Training', epoch=epoch)

            # Validation Phase
            with torch.no_grad():
                val_labels, val_predictions, val_loss = self.run_epoch(progressbar, phase='Validation', epoch=epoch)

            # Compute the metrics, create the logs, and add their plots to tensorboard
            lr = self.optimizer.param_groups[0]['lr']
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
            self.scheduler.step(val_loss)
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

        labels = []
        predictions = []
        total_loss = 0.0
        gradient_norm = 0.0

        for idx, (x_audio, x_text, y) in enumerate(dataset):
            x_audio, x_text, y = x_audio.to(self.device), x_text.to(self.device), y.to(self.device)

            # Forward pass and collect predictions, empty gradients for training
            if backward_pass: self.optimizer.zero_grad()
            output = self.model(x_audio, x_text)
            pred = torch.sigmoid(output).round()
            predictions.append(pred.item())
            labels.append(y.item())

            # Calculate and accumulate loss
            loss = self.criterion(output, y)
            total_loss += loss.item()
            if update_progress_bar:
                n_samples = len(dataset)
                progressbar.set_description(self.get_progressbar_description(phase, epoch, idx, n_samples, loss))

            # Backward pass only for training, and gradient clipping
            if backward_pass:
                loss.backward()
                gradient_norm += self.compute_gradient_norm()
                self.optimizer.step()

            # Free up the memory from GPU
            del x_audio, x_text, y
            torch.cuda.empty_cache()
            gc.collect()

        if backward_pass:
            self.writer.add_scalar(tag='Gradient Norm', scalar_value=gradient_norm, global_step=epoch)
        return labels, predictions, total_loss
