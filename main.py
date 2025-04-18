from src.dataset_downloader import DatasetDownloader
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from src.training import TrainEvalModel
import torch
import click
from dotenv import load_dotenv
import os
import pandas as pd
from itertools import product


class MainClass:
    def __init__(self, imbalance_weighting, device):
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
            'scheduler_patience': 'Int32',
            'stopper_patience': 'Int32',
            'n_epochs': 'Int32'
        })
        self.imbalance_weighting = imbalance_weighting
        self.device = torch.device(device)

    def download(self):
        DatasetDownloader(dataset='DAIC_WoZ', device=self.device)
        DatasetDownloader(dataset='Androids_Corpus', device=self.device)

    def vectorize(self, segment_duration):
        args = {'segment_duration': segment_duration, 'device': self.device}

        # Vectorize and cache Androids-Corpus
        AndroidsCorpusDataset(audio_vectorizer='MelSpec', text_vectorizer='BERT', **args)
        AndroidsCorpusDataset(audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT', **args)
        AndroidsCorpusDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa', **args)

        # Vectorize and cache DAIC-WoZ
        DAICWoZDataset(audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT', **args)
        DAICWoZDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='BERT', **args)
        DAICWoZDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa', **args)

    def single_experiment(self, experiment):
        args = self.experiments[self.experiments.experiment == experiment].iloc[0, 1:].to_dict()
        TrainEvalModel(device=self.device, imbalance_weighting=self.imbalance_weighting, **args)

    def all_experiments(self, dataset):
        audio_vectorizers = ['MelSpec', 'HuBERT', 'Wav2Vec2']
        text_vectorizers = ['BERT', 'ItalianBERT', 'XLMRoBERTa']
        segment_durations = [30, 60]
        args = {
            'dataset': dataset,
            'audio_lstm_hidden_dim': 256,
            'text_lstm_hidden_dim': 256,
            'fc_hidden_dim': 128,
            'lr': 0.00001,
            'weight_decay': 0.00001,
            'scheduler_factor': 0.5,
            'scheduler_patience': 1,
            'stopper_patience': 4,
            'n_epochs': 100,
            'imbalance_weighting': self.imbalance_weighting,
            'device': self.device
        }

        # audio-only modality training
        for audio_vectorizer, segment_duration in product(audio_vectorizers, segment_durations):
            TrainEvalModel(
                audio_vectorizer=audio_vectorizer,
                segment_duration=segment_duration,
                text_vectorizer='BERT',
                modality='audio',
                **args
            )

        # text-only modality training
        for text_vectorizer, segment_duration in product(text_vectorizers, segment_durations):
            TrainEvalModel(
                text_vectorizer=text_vectorizer,
                segment_duration=segment_duration,
                audio_vectorizer='MelSpec',
                modality='text',
                **args
            )

        # Multimodal training
        for audio_vectorizer, text_vectorizer, segment_duration in product(
                audio_vectorizers, text_vectorizers, segment_durations
        ):
            TrainEvalModel(
                audio_vectorizer=audio_vectorizer,
                text_vectorizer=text_vectorizer,
                segment_duration=segment_duration,
                modality='multimodal',
                **args
            )

    def multiple_experiments(self):
        experiments = [
            'AC30_Mel_audio',
            'AC30_HuB_audio',
            'AC30_Wav_audio',
            'AC30_BERT_text',
            'AC30_ITB_text',
            'AC30_RoB_text',
            'AC30_Mel_BERT_multimodal',
            'AC30_Mel_ITB_multimodal',
            'AC30_Mel_RoB_multimodal',
            'AC30_HuB_BERT_multimodal',
            'AC30_HuB_ITB_multimodal',
            'AC30_Wav_BERT_multimodal',
            'AC30_Wav_ITB_multimodal',
            'AC30_Wav_RoB_multimodal',
            'AC60_Mel_audio',
            'AC60_HuB_audio',
            'AC60_Wav_audio',
            'AC60_BERT_text',
            'AC60_ITB_text',
            'AC60_RoB_text',
            'AC60_Mel_BERT_multimodal',
            'AC60_Mel_ITB_multimodal',
            'AC60_Mel_RoB_multimodal',
            'AC60_HuB_BERT_multimodal',
            'AC60_HuB_ITB_multimodal',
            'AC60_Wav_BERT_multimodal',
            'AC60_Wav_ITB_multimodal',
            'AC60_Wav_RoB_multimodal',
            'AC30_HuB_RoB_multimodal',
            'AC60_HuB_RoB_multimodal',
        ]
        for experiment in experiments:
            self.single_experiment(experiment)


@click.command()
@click.option('--download', is_flag=True, help='Download the datasets')
@click.option('--vectorize', is_flag=True, help='Vectorize and cache the datasets')
@click.option('--experiment', help='Which experiment to run from experiments.csv')
@click.option('--all_experiments', help='Run all experiments altogether')
@click.option('--multiple_experiments', is_flag=True, help='Run multiple experiments defined as a list')
@click.option('--device', required=True, help='Select either cuda or cpu to run the experiment')
@click.option('--segment_duration', type=int, help='Select a specific segment duration to split the audio')
def main(download, vectorize, all_experiments, multiple_experiments, experiment, device, segment_duration):
    imbalance_weighting = False
    execute = MainClass(imbalance_weighting=imbalance_weighting, device=device)

    if download:
        execute.download()
    if vectorize:
        execute.vectorize(segment_duration=segment_duration)
    if experiment:
        execute.single_experiment(experiment=experiment)
    if all_experiments:
        execute.all_experiments(dataset=all_experiments)
    if multiple_experiments:
        execute.multiple_experiments()


if __name__ == "__main__":
    """
    This file can be run directly from the command line:
        1- Download the dataset (first run only):
            - python3 main.py --download --device "cuda:0"
        2- Vectorize the dataset (first run only, change the segment duration before running this script):
            - python3 main.py --vectorize --device "cuda:0" --segment_duration 30
        3- Run specific experiment defined in `experiments.csv`:
            - python3 main.py --experiment "AC30_Wav_ITB_multimodal" --device "cuda:0"
        4- Run all experiments:
            - python3 main.py --all_experiments "Androids_Corpus" --device "cuda:0"
        5- Run multiple experiments: 
            - python3 main.py --multiple_experiments --device "cuda:0"
    """
    main()
