from src.feature_extractor import AudioFeatureExtractor, TextFeatureExtractor, LabelTransformer
import os
import pandas as pd
from dotenv import load_dotenv
import librosa
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pickle
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    def __init__(self, raw_data_path, processed_data_path, audio_vectorizer=None, text_vectorizer=None):
        self.RAW_DATA_PATH = raw_data_path
        self.PROCESSED_DATA_PATH = processed_data_path
        self.SAMPLE_RATE = 16000
        self.X_audio = list()
        self.X_text = list()
        self.y = list()
        self.audio_vectorizer = AudioFeatureExtractor(model_name=audio_vectorizer)
        self.text_vectorizer = TextFeatureExtractor(model_name=text_vectorizer)
        self.label_transformer = LabelTransformer()
        self.audio_feature_dim = self.audio_vectorizer.feature_dim
        self.text_feature_dim = self.text_vectorizer.feature_dim

    def cache_dataset(self, cache_name):
        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)
        cache_path = os.path.join(self.PROCESSED_DATA_PATH, cache_name)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.X_audio, self.X_text, self.y = pickle.load(f)
        else:
            self.load_dataset()
            X_audio, X_text, y = self.X_audio, self.X_text, self.y
            self.X_audio, self.X_text, self.y = list(), list(), list()

            for idx in tqdm(range(len(y))):
                self.X_audio.append(self.audio_vectorizer(X_audio[idx]))
                self.X_text.append(self.text_vectorizer(X_text[idx]))
                self.y.append(self.label_transformer(y[idx]))

            with open(cache_path, 'wb') as f:
                pickle.dump((self.X_audio, self.X_text, self.y), f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_audio[idx], self.X_text[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        X_audio, X_text, y = zip(*batch)
        return list(X_audio), list(X_text), list(y)

    @abstractmethod
    def load_dataset(self):
        pass


class DAICWoZDataset(BaseDataset):
    def __init__(self, train_or_dev='train', audio_vectorizer=None, text_vectorizer=None):
        load_dotenv()
        self.DOWNLOAD_ADDRESS = os.getenv('DOWNLOAD_ADDRESS_DAIC_WOZ')
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.train_or_dev = train_or_dev

        super().__init__(
            raw_data_path=os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/DAIC_WoZ'),
            processed_data_path=os.path.join(self.PROJECT_ROOT_PATH, 'data/processed/DAIC_WoZ'),
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer
        )

        self.participants = self.get_participants()
        if audio_vectorizer and text_vectorizer:
            cache_name = f'{audio_vectorizer}_{text_vectorizer}_{train_or_dev}.pkl'
            self.cache_dataset(cache_name)
        else:
            self.load_dataset()

    def get_participants(self):
        """
        [AMHD-GPT]
        The score after filling out the PHQ-8 questionnaire from file 409 is 10. This was
        wrongly listed as not depressive. For this reason, this label is corrected manually.
        """
        excluded_sessions = {342, 394, 398, 460} | {373, 444}
        participants_df = pd.read_csv(f'{self.DOWNLOAD_ADDRESS}/{self.train_or_dev}_split_Depression_AVEC2017.csv')
        participants_df = participants_df[~participants_df['Participant_ID'].isin(excluded_sessions)]
        participants_df.sort_values(by='Participant_ID', inplace=True)
        self.y = (participants_df['PHQ8_Score'] >= 10).astype(int).tolist()
        return participants_df['Participant_ID'].tolist()

    def load_dataset(self):
        for participant in self.participants:
            text_path = os.path.join(self.RAW_DATA_PATH, f'{participant}_TRANSCRIPT.csv')
            interview_df = pd.read_csv(text_path, delimiter='\t')
            interview_df['group'] = (interview_df['speaker'] != interview_df['speaker'].shift()).cumsum()
            interview_df = interview_df.dropna().groupby('group').agg({
                'start_time': 'min',
                'stop_time': 'max',
                'speaker': 'first',
                'value': '. '.join
            }).reset_index(drop=True)
            interview_df = interview_df[interview_df['speaker'] == 'Participant'].copy()
            text_segments = interview_df['value'].tolist()
            self.X_text.append(text_segments)

            audio_path = os.path.join(self.RAW_DATA_PATH, f'{participant}_AUDIO.wav')
            waveform, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
            interview_df['start_time'] = (interview_df['start_time'] * self.SAMPLE_RATE).astype(int)
            interview_df['stop_time'] = (interview_df['stop_time'] * self.SAMPLE_RATE).astype(int)
            audio_segments = [waveform[segment.start_time:segment.stop_time] for segment in interview_df.itertuples()]
            self.X_audio.append(audio_segments)


class AndroidsCorpusDataset(BaseDataset):
    def __init__(self, fold=0, train_or_test='train', audio_vectorizer=None, text_vectorizer=None):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.train_or_test = train_or_test
        self.fold = fold

        super().__init__(
            raw_data_path=os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/Androids_Corpus'),
            processed_data_path=os.path.join(self.PROJECT_ROOT_PATH, 'data/processed/Androids_Corpus'),
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer
        )

        self.participants = self.get_participants()
        if audio_vectorizer and text_vectorizer:
            cache_name = f'{audio_vectorizer}_{text_vectorizer}_{train_or_test}_fold{fold}.pkl'
            self.cache_dataset(cache_name)
        else:
            self.load_dataset()

    def get_participants(self):
        folds_path = os.path.join(self.RAW_DATA_PATH, 'fold-lists.csv')
        folds_df = pd.read_csv(folds_path)
        folds_df = folds_df.iloc[1:, folds_df.columns.tolist().index('Interview'):].T.reset_index(drop=True)
        folds_df = folds_df.apply(lambda col: col.str.strip("'"))

        if self.train_or_test == 'train':
            train_participants = natsorted(folds_df.drop(index=self.fold).stack().dropna().tolist())
            return train_participants
        elif self.train_or_test == 'test':
            test_participants = natsorted(folds_df.iloc[self.fold].dropna().tolist())
            return test_participants

    def load_dataset(self):
        for participant in self.participants:
            depressed = 1 if participant[3] == 'P' else 0
            audio_segments, text_segments = list(), list()
            participant_path = os.path.join(self.RAW_DATA_PATH, participant)

            for data in natsorted(os.listdir(participant_path)):
                data_path = os.path.join(participant_path, data)
                if data.endswith('.wav'):
                    waveform, _ = librosa.load(data_path, sr=self.SAMPLE_RATE)
                    audio_segments.append(waveform)
                elif data.endswith('.txt'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        text_segments.append(f.read())

            self.X_audio.append(audio_segments)
            self.X_text.append(text_segments)
            self.y.append(depressed)
