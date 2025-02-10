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
    def __init__(self, dataset, train_or_test, audio_vectorizer=None, text_vectorizer=None, fold=None):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.PROCESSED_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/processed')
        if dataset == 'DAIC_WoZ':
            self.DOWNLOAD_ADDRESS = os.getenv('DOWNLOAD_ADDRESS_DAIC_WOZ')
            self.RAW_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/DAIC_WoZ')
        elif dataset == 'Androids_Corpus':
            self.RAW_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/Androids_Corpus')
            self.fold = fold

        self.train_or_test = train_or_test
        self.SAMPLE_RATE = 16000
        self.X_audio = list()
        self.X_text = list()
        self.y = list()

        self.participants_df, self.selected_participants, self.train_indices, self.test_indices = \
            self.select_train_test_indices()

        if audio_vectorizer and text_vectorizer:
            self.audio_vectorizer = AudioFeatureExtractor(model_name=audio_vectorizer)
            self.text_vectorizer = TextFeatureExtractor(model_name=text_vectorizer)
            self.label_transformer = LabelTransformer()
            self.audio_feature_dim = self.audio_vectorizer.feature_dim
            self.text_feature_dim = self.text_vectorizer.feature_dim
            self.cache_dataset(cache_name=f'{dataset}_{audio_vectorizer}_{text_vectorizer}.pkl')
        else:
            self.load_dataset()
        self.select_train_test_split()

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

            for idx in tqdm(range(len(y)), desc=f'Caching {cache_name}'):
                self.X_audio.append(self.audio_vectorizer(X_audio[idx]))
                self.X_text.append(self.text_vectorizer(X_text[idx]))
                self.y.append(self.label_transformer(y[idx]))

            with open(cache_path, 'wb') as f:
                pickle.dump((self.X_audio, self.X_text, self.y), f)

    def select_train_test_split(self):
        if self.train_or_test == 'train':
            self.X_audio = [self.X_audio[i] for i in self.train_indices]
            self.X_text = [self.X_text[i] for i in self.train_indices]
            self.y = [self.y[i] for i in self.train_indices]
        elif self.train_or_test == 'test':
            self.X_audio = [self.X_audio[i] for i in self.test_indices]
            self.X_text = [self.X_text[i] for i in self.test_indices]
            self.y = [self.y[i] for i in self.test_indices]

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

    @abstractmethod
    def select_train_test_indices(self):
        pass


class DAICWoZDataset(BaseDataset):
    def __init__(self, train_or_test='train', audio_vectorizer=None, text_vectorizer=None):
        super().__init__(
            dataset='DAIC_WoZ',
            train_or_test=train_or_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer
        )

    def select_train_test_indices(self):
        excluded_sessions = {342, 394, 398, 460} | {373, 444}

        train_df = pd.read_csv(f'{self.DOWNLOAD_ADDRESS}/train_split_Depression_AVEC2017.csv')
        train_df = train_df[~train_df['Participant_ID'].isin(excluded_sessions)]
        train_participants = train_df['Participant_ID'].tolist()

        test_df = pd.read_csv(f'{self.DOWNLOAD_ADDRESS}/dev_split_Depression_AVEC2017.csv')
        test_df = test_df[~test_df['Participant_ID'].isin(excluded_sessions)]
        test_participants = test_df['Participant_ID'].tolist()

        participants_df = pd.concat([train_df, test_df], axis=0, ignore_index=True).sort_values(by='Participant_ID')
        selected_participants = participants_df['Participant_ID'].tolist()
        train_indices = [selected_participants.index(participant) for participant in train_participants]
        test_indices = [selected_participants.index(participant) for participant in test_participants]
        return participants_df, selected_participants, train_indices, test_indices

    def load_dataset(self):
        """
        [AMHD-GPT]
        The score after filling out the PHQ-8 questionnaire from file 409 is 10. This was
        wrongly listed as not depressive. For this reason, this label is corrected manually.
        """
        for participant in self.selected_participants:
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

            PHQ8_score = self.participants_df.loc[self.participants_df['Participant_ID'] == participant, 'PHQ8_Score']
            y = (PHQ8_score >= 10).astype(int).item()
            self.y.append(y)


class AndroidsCorpusDataset(BaseDataset):
    def __init__(self, fold=0, train_or_test='train', audio_vectorizer=None, text_vectorizer=None):
        super().__init__(
            dataset='Androids_Corpus',
            train_or_test=train_or_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer,
            fold=fold
        )

    def select_train_test_split(self):
        participants_df = None
        folds_path = os.path.join(self.RAW_DATA_PATH, 'fold-lists.csv')
        folds_df = pd.read_csv(folds_path)
        folds_df = folds_df.iloc[1:, folds_df.columns.tolist().index('Interview'):].T.reset_index(drop=True)
        folds_df = folds_df.apply(lambda col: col.str.strip("'"))

        train_participants = natsorted(folds_df.drop(index=self.fold).stack().dropna().tolist())
        test_participants = natsorted(folds_df.iloc[self.fold].dropna().tolist())

        selected_participants = natsorted(train_participants + test_participants)
        train_indices = [selected_participants.index(participant) for participant in train_participants]
        test_indices = [selected_participants.index(participant) for participant in test_participants]
        return participants_df, selected_participants, train_indices, test_indices

    def load_dataset(self):
        for participant in self.selected_participants:
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
