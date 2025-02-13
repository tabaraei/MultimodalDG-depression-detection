from src.dataset_downloader import DatasetDownloader
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
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
    # DatasetDownloader(dataset='DAIC_WoZ')
    # DatasetDownloader(dataset='Androids_Corpus')

    # DAICWoZDataset(train_or_test='test', audio_vectorizer='HuBERT', text_vectorizer='BERT')
    # DAICWoZDataset(train_or_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='XLM-RoBERTa')
    # AndroidsCorpusDataset(fold=0, train_or_test='test', audio_vectorizer='HuBERT', text_vectorizer='ItalianBERT')
    # AndroidsCorpusDataset(fold=0, train_or_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='BERT')
    # AndroidsCorpusDataset(fold=0, train_or_test='test', audio_vectorizer='Wav2Vec2', text_vectorizer='XLM-RoBERTa')

    run_experiments(
        dataset='DAIC_WoZ',
        audio_vectorizers=['HuBERT', 'Wav2Vec2'],
        text_vectorizers=['BERT', 'XLM-RoBERTa'],
    )

    run_experiments(
        dataset='Androids_Corpus',
        audio_vectorizers=['HuBERT', 'Wav2Vec2'],
        text_vectorizers=['ItalianBERT', 'BERT', 'XLM-RoBERTa'],
    )
