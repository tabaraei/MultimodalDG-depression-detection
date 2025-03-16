from src.dataset_downloader import DatasetDownloader
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from src.training import TrainEvalModel
import torch
import click
import json
from itertools import product


@click.command()
@click.option('--download', is_flag=True, help='Download the datasets')
@click.option('--vectorize', is_flag=True, help='Vectorize and cache the datasets')
@click.option('--run_all_experiments', help='Run all experiments on either `Androids_Corpus` or `DAIC_WoZ`')
@click.option('--experiment', help='Which experiment to run from experiments.json')
@click.option('--device', required=True, help='Select either cuda or cpu to run the experiment')
def main(download, vectorize, run_all_experiments, experiment, device):
    device = torch.device(device)
    segment_duration = 30

    if download:
        DatasetDownloader(dataset='DAIC_WoZ', device=device)
        DatasetDownloader(dataset='Androids_Corpus', device=device)

    if vectorize:
        print('Vectorize and cache the datasets')
        args = {'segment_duration': segment_duration, 'device': device}

        # Vectorize and cache DAIC-WoZ
        DAICWoZDataset(audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT', **args)
        DAICWoZDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='BERT', **args)
        DAICWoZDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa', **args)

        # Vectorize and cache Androids-Corpus
        AndroidsCorpusDataset(audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT', **args)
        AndroidsCorpusDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='BERT', **args)
        AndroidsCorpusDataset(audio_vectorizer='Wav2Vec2', text_vectorizer='XLMRoBERTa', **args)

    if run_all_experiments:
        dataset = run_all_experiments
        modalities = ['text', 'audio', 'multimodal']
        audio_vectorizers = ['HuBERT', 'Wav2Vec2']
        text_vectorizers = ['XLMRoBERTa']  # ['ItalianBERT', 'BERT']
        for modality, audio_vectorizer, text_vectorizer in product(modalities, audio_vectorizers, text_vectorizers):
            TrainEvalModel(
                dataset=dataset,
                modality=modality,
                audio_vectorizer=audio_vectorizer,
                text_vectorizer=text_vectorizer,
                segment_duration=segment_duration,
                audio_lstm_hidden_dim=256,
                text_lstm_hidden_dim=256,
                fc_hidden_dim=128,
                lr=1e-5 if dataset == 'Androids_Corpus' else 1e-3,
                weight_decay=1e-5,
                scheduler_step_size=20,
                scheduler_gamma=0.5,
                patience=4,
                n_epochs=100,
                device=device
            )

    if experiment:
        with open('experiments.json', 'r') as f:
            experiments = json.load(f)
        TrainEvalModel(device=device, **experiments[experiment])


if __name__ == "__main__":
    """
    This file can be run directly from the command line:
        1- Download the dataset (first run only):
            - python3 main.py --download --device "cuda:0"
        2- Vectorize the dataset (first run only):
            - python3 main.py --vectorize --device "cuda:0"
        3- Run specific experiment defined in `experiments.json`:
            - python3 main.py --experiment "DAIC_HuB_RoB" --device "cuda:0"
        4- Run all experiments: 
            - python3 main.py --run_all_experiments "DAIC_WoZ" --device "cuda:0"
            - python3 main.py --run_all_experiments "Androids_Corpus" --device "cuda:1"
    """
    main()
