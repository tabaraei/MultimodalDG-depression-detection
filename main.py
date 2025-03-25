from src.dataset_downloader import DatasetDownloader
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from src.training import TrainEvalModel
import torch
import click
from dotenv import load_dotenv
import os
import pandas as pd


class MainClass:
    def __init__(self, device):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.EXPERIMENTS_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'experiments.csv')
        self.experiments = pd.read_csv(self.EXPERIMENTS_PATH, delimiter=';', dtype={
            'audio_lstm_hidden_dim': 'Int32',
            'text_lstm_hidden_dim': 'Int32',
            'fc_hidden_dim': 'Int32',
            'lr': 'float64',
            'weight_decay': 'float64',
            'scheduler_factor': 'float64',
            'patience': 'Int32',
            'n_epochs': 'Int32'
        })

        self.device = torch.device(device)
        self.segment_duration = 30

    def download(self):
        DatasetDownloader(dataset='DAIC_WoZ', device=self.device)
        DatasetDownloader(dataset='Androids_Corpus', device=self.device)

    def vectorize(self):
        args = {'segment_duration': self.segment_duration, 'device': self.device}

        # Vectorize and cache Androids-Corpus
        AndroidsCorpusDataset(audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT', **args)
        AndroidsCorpusDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='BERT', **args)
        AndroidsCorpusDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa', **args)

        # Vectorize and cache DAIC-WoZ
        DAICWoZDataset(audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT', **args)
        DAICWoZDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='BERT', **args)
        DAICWoZDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa', **args)

    def single_experiment(self, experiment):
        args = self.experiments[self.experiments.experiment == experiment].iloc[0, 1:14].to_dict()
        TrainEvalModel(device=self.device, **args)

    def multiple_experiments(self):
        experiments = [
            'AC30_HuB_BERT_audio',
            'AC30_HuB_BERT_text',
            'AC30_HuB_BERT_multimodal',
            'AC30_HuB_ITB_audio',
            'AC30_HuB_ITB_text',
            'AC30_HuB_ITB_multimodal',
            'AC30_HuB_RoB_audio',
            'AC30_HuB_RoB_text',
            'AC30_HuB_RoB_multimodal',
            'AC30_Wav_BERT_audio',
            'AC30_Wav_BERT_text',
            'AC30_Wav_BERT_multimodal',
            'AC30_Wav_ITB_audio',
            'AC30_Wav_ITB_text',
            'AC30_Wav_ITB_multimodal',
            'AC30_Wav_RoB_audio',
            'AC30_Wav_RoB_text',
            'AC30_Wav_RoB_multimodal',
            'DAIC30_HuB_BERT_audio',
            'DAIC30_HuB_BERT_text',
            'DAIC30_HuB_BERT_multimodal',
            'DAIC30_HuB_ITB_audio',
            'DAIC30_HuB_ITB_text',
            'DAIC30_HuB_ITB_multimodal',
            'DAIC30_HuB_RoB_audio',
            'DAIC30_HuB_RoB_text',
            'DAIC30_HuB_RoB_multimodal',
            'DAIC30_Wav_BERT_audio',
            'DAIC30_Wav_BERT_text',
            'DAIC30_Wav_BERT_multimodal',
            'DAIC30_Wav_ITB_audio',
            'DAIC30_Wav_ITB_text',
            'DAIC30_Wav_ITB_multimodal',
            'DAIC30_Wav_RoB_audio',
            'DAIC30_Wav_RoB_text',
            'DAIC30_Wav_RoB_multimodal',
        ]
        for experiment in experiments:
            self.single_experiment(experiment)


@click.command()
@click.option('--download', is_flag=True, help='Download the datasets')
@click.option('--vectorize', is_flag=True, help='Vectorize and cache the datasets')
@click.option('--experiment', help='Which experiment to run from experiments.csv')
@click.option('--multiple_experiments', is_flag=True, help='Run multiple experiments defined as a list')
@click.option('--device', required=True, help='Select either cuda or cpu to run the experiment')
def main(download, vectorize, multiple_experiments, experiment, device):
    execute = MainClass(device)
    if download:
        execute.download()
    if vectorize:
        execute.vectorize()
    if experiment:
        execute.single_experiment(experiment)
    if multiple_experiments:
        execute.multiple_experiments()


if __name__ == "__main__":
    """
    This file can be run directly from the command line:
        1- Download the dataset (first run only):
            - python3 main.py --download --device "cuda:0"
        2- Vectorize the dataset (first run only):
            - python3 main.py --vectorize --device "cuda:0"
        3- Run specific experiment defined in `experiments.csv`:
            - python3 main.py --experiment "DAIC30_HuB_BERT_audio" --device "cuda:0"
        4- Run multiple experiments: 
            - python3 main.py --multiple_experiments --device "cuda:0"
    """
    main()
