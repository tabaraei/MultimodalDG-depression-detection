from src.dataset_loader import AndroidsCorpusDataset, DAICWoZDataset
from src.dataset_downloader import DatasetDownloader

import itertools
import torch

if __name__ == "__main__":
    DatasetDownloader(dataset='DAIC_WoZ')
    DatasetDownloader(dataset='Androids_Corpus')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    experiments = itertools.product(range(5), ['train', 'test'], ['HuBERT', 'Wav2Vec2'], ['ItalianBERT', 'BERT'])
    for fold, train_or_test, audio_vectorizer, text_vectorizer in experiments:
        AndroidsCorpusDataset(
            fold=fold,
            train_or_test=train_or_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer,
            device=device
        )
    experiments = itertools.product(['train', 'dev'], ['HuBERT', 'Wav2Vec2'], ['BERT'])
    for train_or_dev, audio_vectorizer, text_vectorizer in experiments:
        DAICWoZDataset(
            train_or_dev=train_or_dev,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer,
            device=device
        )
