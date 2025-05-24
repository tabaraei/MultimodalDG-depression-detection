from src.model import Model
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm.auto import trange
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import os
from dotenv import load_dotenv
from torchinfo import summary
from datetime import datetime
from collections import deque
import shutil
import gc


class EarlyStopping:
    def __init__(self, patience, epsilon=1e-1):
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

        # Early stop due to negligible training loss changes OR due to increasing train/validation loss
        train_diffs_below_epsilon = np.all(abs(np.diff(self.queue_train_losses)) < self.epsilon)
        train_diffs_positive = np.all(np.diff(self.queue_train_losses) > 0)
        val_diffs_positive = np.all(np.diff(self.queue_val_losses) > 0)
        self.early_stop = train_diffs_below_epsilon or train_diffs_positive or val_diffs_positive


class TrainEvalModel:
    def __init__(
            self,
            generalization,
            dataset,
            modality,
            segment_duration,
            audio_vectorizer,
            text_vectorizer,
            audio_lstm_hidden_dim,
            text_lstm_hidden_dim,
            attn_hidden_dim,
            cross_attn_hidden_dim,
            fc_hidden_dim,
            imbalance_weighting,
            lr,
            weight_decay,
            scheduler_factor,
            scheduler_patience,
            stopper_patience,
            lambda_grl,
            n_epochs,
            random_state,
            device,
            idx,
            reset_log_file=True,
            reset_tensorboard=True
    ):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.generalization = generalization
        self.dataset = dataset
        self.modality = modality
        self.segment_duration = segment_duration
        self.audio_vectorizer = audio_vectorizer
        self.text_vectorizer = text_vectorizer
        self.audio_lstm_hidden_dim = audio_lstm_hidden_dim
        self.text_lstm_hidden_dim = text_lstm_hidden_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.cross_attn_hidden_dim = cross_attn_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.imbalance_weighting = imbalance_weighting
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.stopper_patience = stopper_patience
        self.lambda_grl = lambda_grl
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.device = device
        self.idx = idx
        self.reset_log_file = reset_log_file
        self.reset_tensorboard = reset_tensorboard

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.initialize_paths()
        self.initialize_log()
        self.run_pipeline()

    def initialize_paths(self):
        # Set the folder and filenames
        vectorizer = {
            'multimodal': f'{self.audio_vectorizer}_{self.text_vectorizer}',
            'audio': self.audio_vectorizer,
            'text': self.text_vectorizer
        }[self.modality]
        folder_type = 'domain_generalization' if self.generalization else 'normal'
        segment_type = f'_{self.segment_duration}s' if self.segment_duration else ''

        folder_path = f'{folder_type}/{self.dataset}{segment_type}/{self.modality}/{vectorizer}'
        file_name = (
            f'{self.idx}_{self.imbalance_weighting}_{self.audio_lstm_hidden_dim}_{self.text_lstm_hidden_dim}_'
            f'{self.attn_hidden_dim}_{self.cross_attn_hidden_dim}_{self.fc_hidden_dim}_{self.lr}_{self.weight_decay}_'
            f'{self.scheduler_factor}_{self.scheduler_patience}_{self.stopper_patience}_{self.lambda_grl}_{self.n_epochs}'
        )
        self.FILE_PATH = f'{folder_path}/{file_name}'

    def initialize_log(self):
        # Log setup
        self.LOG_FILE_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'logs', f'{self.FILE_PATH}.txt')
        os.makedirs(os.path.dirname(self.LOG_FILE_PATH), exist_ok=True)
        if self.reset_log_file: open(self.LOG_FILE_PATH, 'w').close()

    def initialize_writer(self, fold=None):
        # TensorBoard setup
        fold_type = f'_fold_{fold}' if fold else ''
        self.WRITER_FILE_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'runs', f'{self.FILE_PATH}{fold_type}')
        os.makedirs(os.path.dirname(self.WRITER_FILE_PATH), exist_ok=True)
        if self.reset_tensorboard and os.path.exists(self.WRITER_FILE_PATH):
            shutil.rmtree(self.WRITER_FILE_PATH)
        self.writer = SummaryWriter(self.WRITER_FILE_PATH)

    def log(self, text, print_to_console=True):
        if print_to_console: print(text)
        with open(self.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    def tensorboard_add_model_graph(self):
        with torch.no_grad():
            x_audio, x_text, _ = next(iter(self.train_dataset))
            x_audio, x_text = x_audio.to(self.device), x_text.to(self.device)
            self.writer.add_graph(self.model, input_to_model=(x_audio, x_text))

    def get_progressbar_description(self, phase, epoch, idx, n_samples, loss):
        fold_descr = f'Fold [{self.fold}/{self.n_folds}], ' if self.fold else ''
        return f'{phase}: {fold_descr}' \
               f'Epoch [{epoch + 1}/{self.n_epochs}], ' \
               f'Item [{idx + 1}/{n_samples}], ' \
               f'Loss {loss.item():.3f}'

    def compute_gradient_norm(self):
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def compute_metrics(self, labels, predictions, phase, epoch=None, n_decimals=3):
        accuracy = round(accuracy_score(labels, predictions), n_decimals)
        balanced_accuracy = round(float(balanced_accuracy_score(labels, predictions)), n_decimals)
        precision = round(precision_score(labels, predictions, zero_division=0), n_decimals)
        recall = round(recall_score(labels, predictions, zero_division=0), n_decimals)
        f1 = round(f1_score(labels, predictions, zero_division=0), n_decimals)

        metrics = {
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        self.log(f'{phase} Metrics: {metrics}')
        if phase != 'Test':
            self.writer.add_scalars(main_tag=f'{phase} Metrics', tag_scalar_dict=metrics, global_step=epoch)
        return accuracy, balanced_accuracy, precision, recall, f1

    def clean_GPU_cache(self):
        del self.train_dataset, self.val_dataset, self.test_dataset, self.model
        torch.cuda.empty_cache()
        gc.collect()

    def overcome_GPU_memory_constraints(self):
        if self.audio_vectorizer == 'HuBERT' and self.text_vectorizer == 'XLMRoBERTa':
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
            self.fold = None
            self.initialize_writer()
            self.train_dataset = DAICWoZDataset(train_val_test='train', **args)
            self.val_dataset = DAICWoZDataset(train_val_test='val', **args)
            self.test_dataset = DAICWoZDataset(train_val_test='test', **args)
            self.train_and_evaluate()
            self.writer.close()
            self.clean_GPU_cache()

        elif self.dataset == 'Androids_Corpus':
            fold_metrics = list()
            self.n_folds = 5
            for fold in range(self.n_folds):
                self.fold = fold + 1
                self.initialize_writer(fold=fold)
                args_fold = {'fold': fold, 'random_state': self.random_state, **args}
                self.train_dataset = AndroidsCorpusDataset(train_val_test='train', **args_fold)
                self.val_dataset = AndroidsCorpusDataset(train_val_test='val', **args_fold)
                self.test_dataset = AndroidsCorpusDataset(train_val_test='test', **args_fold)
                accuracy, balanced_accuracy, precision, recall, f1 = self.train_and_evaluate()
                fold_metrics.append([accuracy, balanced_accuracy, precision, recall, f1])
                self.writer.close()
                self.clean_GPU_cache()

            accuracy, balanced_accuracy, precision, recall, f1 = np.mean(fold_metrics, axis=0)
            accuracy_std, balanced_accuracy_std, precision_std, recall_std, f1_std = np.std(fold_metrics, axis=0)
            self.log(f"{'=' * 14} 5-fold Cross Validation Test Results {'=' * 13}")
            self.log(
                f'Accuracy: {accuracy * 100:.2f}% ± {accuracy_std * 100:.2f}%\n'
                f'Balanced Accuracy: {balanced_accuracy * 100:.2f}% ± {balanced_accuracy_std * 100:.2f}%\n'
                f'Precision: {precision * 100:.2f}% ± {precision_std * 100:.2f}%\n'
                f'Recall: {recall * 100:.2f}% ± {recall_std * 100:.2f}%\n'
                f'F1-Score: {f1 * 100:.2f}% ± {f1_std * 100:.2f}%'
            )

        end_time = datetime.now()
        self.log(f'Pipeline Execution Time: {str(end_time - start_time).split('.')[0]}')

    def train_and_evaluate(self):
        # Model initialization
        if self.imbalance_weighting:
            pos_weight = (len(self.train_dataset) - sum(self.train_dataset.y)) / sum(self.train_dataset.y)
            self.depression_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        else:
            self.depression_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.model = Model(
            generalization=self.generalization,
            modality=self.modality,
            audio_feature_dim=self.train_dataset.audio_feature_dim,
            text_feature_dim=self.train_dataset.text_feature_dim,
            audio_lstm_hidden_dim=self.audio_lstm_hidden_dim,
            text_lstm_hidden_dim=self.text_lstm_hidden_dim,
            attn_hidden_dim=self.attn_hidden_dim,
            cross_attn_hidden_dim=self.cross_attn_hidden_dim,
            fc_hidden_dim=self.fc_hidden_dim,
            n_domains=len(self.train_dataset),
            lambda_grl=self.lambda_grl
        ).to(self.device)
        self.log(str(summary(self.model)), print_to_console=False)
        self.tensorboard_add_model_graph()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.stopper = EarlyStopping(patience=self.stopper_patience)
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            threshold_mode='abs',
            threshold=1e-3
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
            fold_descr = f'Fold [{self.fold}/{self.n_folds}], ' if self.fold else ''
            self.log(f"{fold_descr}Epoch [{epoch + 1}/{self.n_epochs}]:")
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
            update_progress_bar = True
        elif phase == 'Validation':
            self.model.eval()
            dataset = self.val_dataset
            update_progress_bar = True
        elif phase == 'Test':
            self.model.eval()
            dataset = self.test_dataset
            update_progress_bar = False

        labels = []
        predictions = []
        total_loss = 0.0
        gradient_norm = 0.0

        for idx, (x_audio, x_text, y) in enumerate(dataset):
            # Duplicate labels and domains across segments, move tensors to device, empty gradients for training
            n_segments = x_audio.shape[0]
            depression_labels = torch.full((n_segments, 1), y.item(), dtype=torch.float32).to(self.device)
            domain_labels = torch.full((n_segments,), idx, dtype=torch.long).to(self.device)
            x_audio, x_text = x_audio.to(self.device), x_text.to(self.device)
            if self.model.training: self.optimizer.zero_grad()

            # Forward pass, calculate loss
            if self.model.training and self.generalization:
                depression_logits, domain_logits = self.model(x_audio, x_text)
                depression_loss = self.depression_criterion(depression_logits, depression_labels)
                domain_loss = self.domain_criterion(domain_logits, domain_labels)
                loss = depression_loss + domain_loss
            else:
                depression_logits = self.model(x_audio, x_text)
                depression_loss = self.depression_criterion(depression_logits, depression_labels)
                loss = depression_loss

            # Accumulate loss
            total_loss += loss.item()
            if update_progress_bar:
                n_samples = len(dataset)
                progressbar.set_description(self.get_progressbar_description(phase, epoch, idx, n_samples, loss))

            # Collect predictions and by taking average logit
            pred = torch.sigmoid(depression_logits.mean()).round().item()
            predictions.append(pred)
            labels.append(y.item())

            # Backward pass only for training, and gradient clipping
            if self.model.training:
                loss.backward()
                gradient_norm += self.compute_gradient_norm()
                self.optimizer.step()

            # Free up the memory from GPU
            del x_audio, x_text, y, depression_labels, domain_labels
            torch.cuda.empty_cache()
            gc.collect()

        if self.model.training:
            self.writer.add_scalar(tag='Gradient Norm', scalar_value=gradient_norm, global_step=epoch)
        return labels, predictions, total_loss
