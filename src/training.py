from src.model import MultimodalClassifier
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from torch.optim.lr_scheduler import StepLR
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
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
    ):
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
        self.create_log_file(reset_log_file)
        self.run_pipeline()

    def create_log_file(self, reset_log_file):
        load_dotenv()
        PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        LOG_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data/log')
        os.makedirs(LOG_DATA_PATH, exist_ok=True)

        self.LOG_FILE_PATH = (
            f'{LOG_DATA_PATH}/'
            f'{self.dataset}_{self.audio_vectorizer}_{self.text_vectorizer}_'
            f'{self.lstm_hidden_dim}_{self.fc_hidden_dim}.txt'
        )
        if reset_log_file:
            open(self.LOG_FILE_PATH, 'w').close()

    def log(self, text):
        print(text)
        with open(self.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    def compute_metrics(self, labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        self.log(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
        self.log('-' * 65)
        return accuracy, precision, recall, f1

    def run_pipeline(self):
        start_time = datetime.now()
        vectorizers = {'audio_vectorizer': self.audio_vectorizer, 'text_vectorizer': self.text_vectorizer}

        if self.dataset == 'DAIC_WoZ':
            self.train_dataset = DAICWoZDataset(train_val_test='train', **vectorizers)
            self.val_dataset = DAICWoZDataset(train_val_test='val', **vectorizers)
            self.test_dataset = DAICWoZDataset(train_val_test='test', **vectorizers)
            self.train_and_evaluate()

        elif self.dataset == 'Androids_Corpus':
            fold_metrics = list()
            for fold in range(5):
                self.train_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='train', **vectorizers)
                self.val_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='val', **vectorizers)
                self.test_dataset = AndroidsCorpusDataset(fold=fold, train_val_test='test', **vectorizers)

                accuracy, precision, recall, f1 = self.train_and_evaluate()
                fold_metrics.append([accuracy, precision, recall, f1])

            accuracy, precision, recall, f1 = np.mean(fold_metrics, axis=0)
            self.log(f"{'=' * 14} 5-fold Cross Validation Test Results {'=' * 13}")
            self.log(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

        end_time = datetime.now()
        self.log(f'Pipeline Execution Time: {str(end_time - start_time).split('.')[0]}')

    def train_and_evaluate(self):
        # Model initialization
        model = MultimodalClassifier(
            audio_feature_dim=self.train_dataset.audio_feature_dim,
            text_feature_dim=self.train_dataset.text_feature_dim,
            lstm_n_layers=self.lstm_n_layers,
            lstm_hidden_dim=self.lstm_hidden_dim,
            fc_hidden_dim=self.fc_hidden_dim
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        self.log(str(summary(model)))

        # Training loop over epochs
        epochs = trange(self.n_epochs)
        for epoch in epochs:
            # --------------------------------- Training ---------------------------------
            model.train()
            training_loss = 0.0

            for idx, (x_audio, x_text, y) in enumerate(self.train_dataset):
                x_audio = x_audio.to(self.device)
                x_text = x_text.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = model(x_audio, x_text).squeeze(1).cpu()
                loss = criterion(output, y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                epochs.set_description(
                    f'TRAINING: Epoch [{epoch + 1}/{self.n_epochs}], '
                    f'Item [{idx + 1}/{len(self.train_dataset)}], '
                    f'Loss {loss.item():.3f}'
                )
                training_loss += loss.item()

            # -------------------------------- Validation --------------------------------
            model.eval()
            validation_loss = 0.0
            val_predictions = []
            val_ground_truths = []

            with torch.no_grad():
                for idx, (x_audio, x_text, y) in enumerate(self.val_dataset):
                    x_audio = x_audio.to(self.device)
                    x_text = x_text.to(self.device)

                    # Forward pass and collect predictions
                    output = model(x_audio, x_text).squeeze(1).cpu()
                    loss = criterion(output, y)
                    pred = torch.sigmoid(output).round()
                    val_predictions.append(pred.item())
                    val_ground_truths.append(y.item())

                    # Accumulate loss
                    epochs.set_description(
                        f'VALIDATION: Epoch [{epoch + 1}/{self.n_epochs}], '
                        f'Item [{idx + 1}/{len(self.val_dataset)}], '
                        f'Loss {loss.item():.3f}'
                    )
                    validation_loss += loss.item()

            # Compute metrics on the validation set
            self.log(f'Epoch [{epoch + 1}/{self.n_epochs}]:')
            self.log(f'Training Loss: {training_loss:.3f}, Validation Loss: {validation_loss:.3f}')
            self.compute_metrics(val_ground_truths, val_predictions)

            scheduler.step()

        # ---------------------------------- Evaluation ----------------------------------
        model.eval()
        test_predictions = []
        test_ground_truths = []

        with torch.no_grad():
            for idx, (x_audio, x_text, y) in enumerate(self.test_dataset):
                x_audio = x_audio.to(self.device)
                x_text = x_text.to(self.device)

                # Forward pass and collect predictions
                output = model(x_audio, x_text).squeeze(1).cpu()
                pred = torch.sigmoid(output).round()
                test_predictions.append(pred.item())
                test_ground_truths.append(y.item())

        # Compute metrics on the validation set
        self.log(f"{'=' * 26} Test Results {'=' * 25}")
        accuracy, precision, recall, f1 = self.compute_metrics(test_ground_truths, test_predictions)

        return accuracy, precision, recall, f1
