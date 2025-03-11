from src.dataset_downloader import DatasetDownloader
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from src.training import TrainEvalModel
import click
import json
from itertools import product


@click.command()
@click.option('--download', is_flag=True, help='Download the datasets')
@click.option('--vectorize', is_flag=True, help='Vectorize and cache the datasets')
@click.option('--run_all_experiments', help='Run all experiments on either `Androids_Corpus` or `DAIC_WoZ`')
@click.option('--experiment', help='Which experiment to run from experiments.json')
def main(download, vectorize, run_all_experiments, experiment):
    if download:
        DatasetDownloader(dataset='DAIC_WoZ')
        DatasetDownloader(dataset='Androids_Corpus')

    if vectorize:
        # Vectorize and cache DAIC-WoZ
        DAICWoZDataset(train_val_test='test', audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT')
        DAICWoZDataset(train_val_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='BERT')
        DAICWoZDataset(train_val_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa')

        # Vectorize and cache Androids-Corpus
        AndroidsCorpusDataset(fold=0, train_val_test='test', audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT')
        AndroidsCorpusDataset(fold=0, train_val_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='BERT')
        AndroidsCorpusDataset(fold=0, train_val_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa')

    if run_all_experiments:
        dataset = run_all_experiments
        audio_vectorizers = ['HuBERT', 'Wav2Vec2']
        text_vectorizers = ['ItalianBERT', 'BERT', 'XLMRoBERTa']
        for audio_vectorizer, text_vectorizer in product(audio_vectorizers, text_vectorizers):
            TrainEvalModel(
                dataset=dataset,
                modality='multimodal',
                audio_vectorizer=audio_vectorizer,
                text_vectorizer=text_vectorizer,
                audio_lstm_hidden_dim=256,
                text_lstm_hidden_dim=256,
                fc_hidden_dim=128,
                lr=0.0005 if dataset == 'Androids_Corpus' else 0.001,
                weight_decay=1e-6,
                scheduler_step_size=25,
                scheduler_gamma=0.5,
                patience=4,
                n_epochs=100,
                device='cuda:0'
            )

    if experiment:
        with open('experiments.json', 'r') as f:
            experiments = json.load(f)
        TrainEvalModel(**experiments[experiment])


if __name__ == "__main__":
    """
    This file can be run directly from the command line:
        1- Download the dataset (first run only): python3 main.py --download
        2- Vectorize the dataset (first run only): python3 main.py --vectorize
        3- Run specific experiment defined in `experiments.json`: python3 main.py --experiment "DAIC_HuB_RoB"
        4- Run all experiments: 
            - python3 main.py --run_all_experiments "DAIC_WoZ"
            - python3 main.py --run_all_experiments "Androids_Corpus"
    """
    main()
