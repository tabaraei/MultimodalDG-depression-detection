from src.dataset_downloader import DatasetDownloader
from src.training import TrainEvalModel
import itertools


def run_experiments(dataset, audio_vectorizers, text_vectorizers):
    experiments = itertools.product(audio_vectorizers, text_vectorizers)
    for audio_vectorizer, text_vectorizer in experiments:
        TrainEvalModel(
            dataset=dataset,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer,
            reset_log_file=True
        )


if __name__ == "__main__":
    DatasetDownloader(dataset='DAIC_WoZ')
    DatasetDownloader(dataset='Androids_Corpus')
    run_experiments(
        dataset='DAIC_WoZ',
        audio_vectorizers=['HuBERT', 'Wav2Vec2'],
        text_vectorizers=['BERT'],
    )
    run_experiments(
        dataset='Androids_Corpus',
        audio_vectorizers=['HuBERT', 'Wav2Vec2'],
        text_vectorizers=['ItalianBERT', 'BERT'],
    )
