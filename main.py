from src.dataset_downloader import DatasetDownloader
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from src.training import TrainEvalModel
import click
import json


@click.command()
@click.option('--download', is_flag=True, help='Download the dataset')
@click.option('--vectorize', is_flag=True, help='Vectorize and cache the dataset')
@click.option('--experiment', default='AC_Wav_RoB', help='Which experiment to run from experiments.json')
def main(download, vectorize, experiment):
    if download:
        DatasetDownloader(dataset='DAIC_WoZ')
        DatasetDownloader(dataset='Androids_Corpus')

    if vectorize:
        # Vectorize and cache DAIC-WoZ
        DAICWoZDataset(train_or_test='test', audio_vectorizer='HuBERT', text_vectorizer='BERT')
        DAICWoZDataset(train_or_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='XLM-RoBERTa')

        # Vectorize and cache Androids-Corpus
        AndroidsCorpusDataset(fold=0, train_or_test='test', audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT')
        AndroidsCorpusDataset(fold=0, train_or_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='BERT')
        AndroidsCorpusDataset(fold=0, train_or_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='XLM-RoBERTa')

    if experiment:
        with open('experiments.json', 'r') as f:
            experiments = json.load(f)
        TrainEvalModel(**experiments[experiment])


if __name__ == "__main__":
    """
    This file can be run directly from the command line:
        1- Download and vectorize (first run only): python3 main.py --download --vectorize
        2- Experiment: python3 main.py --experiment "AC_Wav_RoB"
    """
    main()
