from src.model import MultimodalClassifier
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm.auto import trange
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from dotenv import load_dotenv


class TrainEvalModel:
    def __init__(
            self,
            dataset,
            audio_vectorizer,
            text_vectorizer,
            lstm_n_layers=1,
            lstm_hidden_dim=256,
            fc_hidden_dim=128,
            n_epochs=100,
            lr=0.001,
            reset_log_file=True
    ):
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.audio_vectorizer = audio_vectorizer
        self.text_vectorizer = text_vectorizer
        self.lstm_n_layers = lstm_n_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss()
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

    def run_pipeline(self):
        if self.dataset == 'DAIC_WoZ':
            self.train_dataset = DAICWoZDataset(
                train_or_dev='train',
                audio_vectorizer=self.audio_vectorizer,
                text_vectorizer=self.text_vectorizer
            )
            self.test_dataset = DAICWoZDataset(
                train_or_dev='dev',
                audio_vectorizer=self.audio_vectorizer,
                text_vectorizer=self.text_vectorizer
            )
            accuracy, precision, recall, f1 = self.train_and_evaluate()
            self.log(f"{'=' * 42} Test Results {'=' * 42}")
            self.log(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

        elif self.dataset == 'Androids_Corpus':
            fold_metrics = list()
            for fold in range(5):
                self.train_dataset = AndroidsCorpusDataset(
                    fold=fold,
                    train_or_test='train',
                    audio_vectorizer=self.audio_vectorizer,
                    text_vectorizer=self.text_vectorizer
                )
                self.test_dataset = AndroidsCorpusDataset(
                    fold=fold,
                    train_or_test='test',
                    audio_vectorizer=self.audio_vectorizer,
                    text_vectorizer=self.text_vectorizer
                )
                accuracy, precision, recall, f1 = self.train_and_evaluate()
                fold_metrics.append([accuracy, precision, recall, f1])
            accuracy, precision, recall, f1 = np.mean(fold_metrics, axis=0)
            self.log(f"{'=' * 30} 5-fold Cross Validation Test Results {'=' * 30}")
            self.log(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

    def train_and_evaluate(self):
        # Model initialization
        self.model = MultimodalClassifier(
            audio_feature_dim=self.train_dataset.audio_feature_dim,
            text_feature_dim=self.train_dataset.text_feature_dim,
            lstm_n_layers=self.lstm_n_layers,
            lstm_hidden_dim=self.lstm_hidden_dim,
            fc_hidden_dim=self.fc_hidden_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Loop over epochs
        epochs = trange(self.n_epochs)
        for epoch in epochs:
            # --------------------------------- Training ---------------------------------
            self.model.train()
            training_loss = 0.0

            for idx, (x_audio, x_text, y) in enumerate(self.train_dataset):
                x_audio = x_audio.to(self.device)
                x_text = x_text.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(x_audio, x_text).squeeze(1)
                loss = self.criterion(output, y)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                epochs.set_description(
                    f'TRAINING: Epoch [{epoch + 1}/{self.n_epochs}], '
                    f'Item [{idx + 1}/{len(self.train_dataset)}], '
                    f'Loss {loss.item():.3f}'
                )
                training_loss += loss.item()

            # -------------------------------- Evaluation --------------------------------
            self.model.eval()
            test_loss = 0.0
            predictions = []
            ground_truths = []

            with torch.no_grad():
                for idx, (x_audio, x_text, y) in enumerate(self.test_dataset):
                    x_audio = x_audio.to(self.device)
                    x_text = x_text.to(self.device)

                    # Forward pass and collect predictions
                    output = self.model(x_audio, x_text).squeeze(1)
                    loss = self.criterion(output, y)
                    pred = torch.sigmoid(output).round()
                    predictions.append(pred.item())
                    ground_truths.append(y.item())

                    # Accumulate loss
                    epochs.set_description(
                        f'EVALUATION: Epoch [{epoch + 1}/{self.n_epochs}], '
                        f'Item [{idx + 1}/{len(self.test_dataset)}], '
                        f'Loss {loss.item():.3f}'
                    )
                    test_loss += loss.item()

            # Compute metrics
            accuracy = accuracy_score(ground_truths, predictions)
            precision = precision_score(ground_truths, predictions, zero_division=0)
            recall = recall_score(ground_truths, predictions, zero_division=0)
            f1 = f1_score(ground_truths, predictions, zero_division=0)

            self.log(f'Epoch [{epoch + 1}/{self.n_epochs}]:')
            self.log(f'Training Loss: {training_loss:.3f}, Validation Loss: {test_loss:.3f}')
            self.log(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
            self.log('-' * 80)

        return accuracy, precision, recall, f1
