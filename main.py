from src.dataset_downloader import DatasetDownloader
from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
from src.training import TrainEvalModel
import torch
import click
from dotenv import load_dotenv
import os
from itertools import product


class MainClass:
    def __init__(self, dataset, modality, generalization, device):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.RAW_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, f'data/raw/{dataset}')
        self.PROCESSED_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, f'data/processed')
        self.dataset = dataset
        self.modality = modality
        self.generalization = generalization
        self.device = torch.device(device)
        self.segment_durations = [20, 30, 45, 60]
        self.audio_vectorizers = ['MelSpec', 'HuBERT', 'Wav2Vec2']
        self.text_vectorizers = ['BERT', 'ItalianBERT', 'XLMRoBERTa']
        self.download_and_vectorize_dataset()

        self.args = {
            'dataset': self.dataset,
            'audio_lstm_hidden_dim': 256,
            'text_lstm_hidden_dim': 256,
            'attn_hidden_dim': 128,
            'cross_attn_hidden_dim': 256,
            'fc_hidden_dim': 128,
            'lr': 0.00001,
            'weight_decay': 0.00001,
            'scheduler_factor': 0.3,
            'scheduler_patience': 2,
            'stopper_patience': 4,
            'n_epochs': 100,
            'imbalance_weighting': False,
            'random_state': 21,
            'device': self.device
        }

    def download_and_vectorize_dataset(self):
        """
        Extracts and caches the features and labels into a dedicated folder for further use during training.
        This function is only run once in the initial phase after downloading the dataset.
        """
        if not os.path.exists(self.RAW_DATA_PATH):
            DatasetDownloader(dataset=self.dataset, device=self.device)

        for segment_duration in self.segment_durations:
            label_file_name = f"{self.dataset}_{f'{segment_duration}s_' if segment_duration else ''}label.pkl"
            label_path = os.path.join(self.PROCESSED_DATA_PATH, label_file_name)
            if not os.path.exists(label_path):
                args = {'segment_duration': segment_duration, 'device': self.device}
                Dataset = AndroidsCorpusDataset if self.dataset == 'Androids_Corpus' else DAICWoZDataset
                for audio_vectorizer, text_vectorizer in zip(self.audio_vectorizers, self.text_vectorizers):
                    Dataset(audio_vectorizer=audio_vectorizer, text_vectorizer=text_vectorizer, **args)

    def single_experiment(self, experiment):
        """
        Valid modalities to be used for this function are "text", "audio", and "multimodal"
        Expected experiment name is as follows: "{audio_vectorizer}_{text_vectorizer}_{segment_duration}"
        where segment_duration is in seconds and can be any integer number.
        Examples:
            - "MelSpec_BERT_30"
            - "MelSpec_ItalianBERT_30"
            - "MelSpec_XLMRoBERTa_30"
            - "HuBERT_BERT_30"
            - "HuBERT_ItalianBERT_30"
            - "HuBERT_XLMRoBERTa_30"
            - "Wav2Vec2_BERT_30"
            - "Wav2Vec2_ItalianBERT_30"
            - "Wav2Vec2_XLMRoBERTa_30"
        """
        audio_vectorizer, text_vectorizer, segment_duration = experiment.split("_")
        TrainEvalModel(
            generalization=self.generalization,
            modality=self.modality,
            segment_duration=segment_duration,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer,
            idx=0,
            **self.args
        )

    def all_experiments(self, n_repeats=5):
        """
        Valid modalities to be used for this function are "text", "audio", "multimodal", and "all"
        In case "all" is selected, all the modalities will be considered for the execution of the experiments
        This function runs all combination of vectorizers and given segment durations for n_repeats times.
        Different experiments can be run in parallel over different GPUs by modifying segment_duration
        Note that XLMRoBERTa is very slow, and it is suggested to be excluded below, and run separately on CPU
        """
        args = self.args.copy()
        for i in range(n_repeats):
            args['idx'] = i + 1

            # audio-only modality training
            if self.modality in ['audio', 'all']:
                for audio_vectorizer, segment_duration in product(self.audio_vectorizers, self.segment_durations):
                    TrainEvalModel(
                        generalization=self.generalization,
                        segment_duration=segment_duration,
                        audio_vectorizer=audio_vectorizer,
                        text_vectorizer='BERT',
                        modality='audio',
                        **args
                    )

            # text-only modality training
            if self.modality in ['text', 'all']:
                for text_vectorizer, segment_duration in product(self.text_vectorizers, self.segment_durations):
                    TrainEvalModel(
                        generalization=self.generalization,
                        segment_duration=segment_duration,
                        text_vectorizer=text_vectorizer,
                        audio_vectorizer='MelSpec',
                        modality='text',
                        **args
                    )

            # Multimodal training
            if self.modality in ['multimodal', 'all']:
                for audio_vectorizer, text_vectorizer, segment_duration in product(
                        self.audio_vectorizers, self.text_vectorizers, self.segment_durations
                ):
                    TrainEvalModel(
                        generalization=self.generalization,
                        segment_duration=segment_duration,
                        audio_vectorizer=audio_vectorizer,
                        text_vectorizer=text_vectorizer,
                        modality='multimodal',
                        **args
                    )


@click.command()
@click.option('--dataset', required=True, help='Dataset name, either "Androids_Corpus" or "DAIC_WoZ"')
@click.option('--modality', help='Which modality for experiments: "text", "audio", "multimodal", or "all"')
@click.option('--device', required=True, help='Select from "cuda:0", "cuda:1", or "cpu" for the execution')
@click.option('--experiment', help='Run only a single specified experiment')
@click.option('--all_experiments', help='Run all experiments altogether')
@click.option('--generalization', is_flag=True, help='Activate domain generalization if included')
def main(dataset, modality, device, experiment, all_experiments, generalization):
    execute = MainClass(dataset=dataset, modality=modality, generalization=generalization, device=device)
    if experiment:
        execute.single_experiment(experiment=experiment)
    if all_experiments:
        execute.all_experiments()


if __name__ == "__main__":
    """
    This file can be run directly from the command line for the "Androids_Corpus" or "DAIC_WoZ" dataset:
    Run specific experiment:
        - With domain generalization: 
            python3 main.py --experiment "MelSpec_ItalianBERT_30" --dataset "Androids_Corpus" --modality "multimodal" --device "cuda:0" --generalization
        - Without domain generalization: 
            python3 main.py --experiment "MelSpec_ItalianBERT_30" --dataset "Androids_Corpus" --modality "multimodal" --device "cuda:0"
    Run all experiments:
        - With domain generalization: 
            python3 main.py --all_experiments --dataset "Androids_Corpus" --modality "multimodal" --device "cuda:0" --generalization
        - Without domain generalization: 
            python3 main.py --all_experiments --dataset "Androids_Corpus" --modality "multimodal" --device "cuda:0"
    """
    main()
