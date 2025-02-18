from src.feature_extractor import AudioFeatureExtractor, TextFeatureExtractor, LabelTransformer
import os
import pandas as pd
from dotenv import load_dotenv
import librosa
from natsort import natsorted
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pickle
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    def __init__(self, dataset, train_val_test, audio_vectorizer=None, text_vectorizer=None, fold=None):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.PROCESSED_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/processed')
        self.train_val_test = train_val_test
        self.SAMPLE_RATE = 16000
        self.dataset = dataset
        if self.dataset == 'DAIC_WoZ':
            self.DOWNLOAD_ADDRESS = os.getenv('DOWNLOAD_ADDRESS_DAIC_WOZ')
            self.RAW_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/DAIC_WoZ')
        elif self.dataset == 'Androids_Corpus':
            self.RAW_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/Androids_Corpus')
            self.fold = fold

        self.participants_df, self.selected_participants, self.train_indices, self.val_indices, self.test_indices = \
            self.select_indices()

        if audio_vectorizer and text_vectorizer:
            self.audio_vectorizer = AudioFeatureExtractor(model_name=audio_vectorizer)
            self.text_vectorizer = TextFeatureExtractor(model_name=text_vectorizer)
            self.label_transformer = LabelTransformer()

            self.audio_feature_dim = self.audio_vectorizer.feature_dim
            self.text_feature_dim = self.text_vectorizer.feature_dim

            self.X_audio = self.load_cache(data_type='audio', name=audio_vectorizer, transformer=self.audio_vectorizer)
            self.X_text = self.load_cache(data_type='text', name=text_vectorizer, transformer=self.text_vectorizer)
            self.y = self.load_cache(data_type='label', name='label', transformer=self.label_transformer)
        else:
            self.X_audio = self.load_raw_data(data_type='audio')
            self.X_text = self.load_raw_data(data_type='text')
            self.y = self.load_raw_data(data_type='label')
        self.select_train_test_split()

    def load_cache(self, data_type, name, transformer):
        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)
        cache_name = f'{self.dataset}_{name}.pkl'
        cache_path = os.path.join(self.PROCESSED_DATA_PATH, cache_name)
        cache_data = list()

        if not os.path.exists(cache_path):
            raw_data = self.load_raw_data(data_type=data_type)
            with open(cache_path, 'ab+') as f:
                for idx in tqdm(range(len(self.selected_participants)), desc=f'Caching {cache_name}'):
                    pickle.dump(transformer(raw_data[idx]), f)
            del raw_data

        with open(cache_path, 'rb') as f:
            for idx in range(len(self.selected_participants)):
                cache_data.append(pickle.load(f))
        return cache_data

    def select_train_test_split(self):
        if self.train_val_test == 'train':
            indices = self.train_indices
        elif self.train_val_test == 'val':
            indices = self.val_indices
        elif self.train_val_test == 'test':
            indices = self.test_indices

        self.X_audio = [self.X_audio[i] for i in indices]
        self.X_text = [self.X_text[i] for i in indices]
        self.y = [self.y[i] for i in indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_audio[idx], self.X_text[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        X_audio, X_text, y = zip(*batch)
        return list(X_audio), list(X_text), list(y)

    @abstractmethod
    def load_raw_data(self, data_type):
        pass

    @abstractmethod
    def select_indices(self):
        pass


class DAICWoZDataset(BaseDataset):
    def __init__(self, train_val_test='train', audio_vectorizer=None, text_vectorizer=None):
        super().__init__(
            dataset='DAIC_WoZ',
            train_val_test=train_val_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer
        )

    def select_indices(self):
        excluded_sessions = {342, 394, 398, 460} | {373, 444}

        train_df = pd.read_csv(f'{os.path.join(self.RAW_DATA_PATH, "train_df.csv")}')
        train_df = train_df[~train_df['Participant_ID'].isin(excluded_sessions)]
        train_participants = train_df['Participant_ID'].tolist()

        val_df = pd.read_csv(f'{os.path.join(self.RAW_DATA_PATH, "val_df.csv")}')
        val_df = val_df[~val_df['Participant_ID'].isin(excluded_sessions)]
        val_participants = val_df['Participant_ID'].tolist()

        test_df = pd.read_csv(f'{os.path.join(self.RAW_DATA_PATH, "test_df.csv")}')
        test_df = test_df[~test_df['Participant_ID'].isin(excluded_sessions)]
        test_df.rename(columns={'PHQ_Score': 'PHQ8_Score', 'PHQ_Binary': 'PHQ8_Binary'}, inplace=True)
        test_participants = test_df['Participant_ID'].tolist()

        participants_df = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values(by='Participant_ID')
        selected_participants = participants_df['Participant_ID'].tolist()
        train_indices = [selected_participants.index(participant) for participant in train_participants]
        val_indices = [selected_participants.index(participant) for participant in val_participants]
        test_indices = [selected_participants.index(participant) for participant in test_participants]

        return participants_df, selected_participants, train_indices, val_indices, test_indices

    def load_raw_data(self, data_type):
        """
        [AMHD-GPT]
        The score after filling out the PHQ-8 questionnaire from file 409 is 10. This was
        wrongly listed as not depressive. For this reason, this label is corrected manually.
        """
        raw_data = list()
        for participant in self.selected_participants:
            if data_type == 'label':
                PHQ8_score = self.participants_df.loc[
                    self.participants_df['Participant_ID'] == participant, 'PHQ8_Score']
                y = (PHQ8_score >= 10).astype(int).item()
                raw_data.append(y)
            else:
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

                if data_type == 'text':
                    text_segments = interview_df['value'].tolist()
                    raw_data.append(text_segments)
                elif data_type == 'audio':
                    audio_path = os.path.join(self.RAW_DATA_PATH, f'{participant}_AUDIO.wav')
                    waveform, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
                    interview_df['start_time'] = (interview_df['start_time'] * self.SAMPLE_RATE).astype(int)
                    interview_df['stop_time'] = (interview_df['stop_time'] * self.SAMPLE_RATE).astype(int)
                    audio_segments = [waveform[segment.start_time:segment.stop_time] for segment in
                                      interview_df.itertuples()]
                    raw_data.append(audio_segments)

        return raw_data


class AndroidsCorpusDataset(BaseDataset):
    def __init__(self, fold=0, train_val_test='train', audio_vectorizer=None, text_vectorizer=None):
        super().__init__(
            dataset='Androids_Corpus',
            train_val_test=train_val_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer,
            fold=fold
        )

    def select_indices(self):
        participants_df = None
        folds_path = os.path.join(self.RAW_DATA_PATH, 'fold-lists.csv')
        folds_df = pd.read_csv(folds_path)
        folds_df = folds_df.iloc[1:, folds_df.columns.tolist().index('Interview'):].T.reset_index(drop=True)
        folds_df = folds_df.apply(lambda col: col.str.strip("'"))

        train_participants = natsorted(folds_df.drop(index=self.fold).stack().dropna().tolist())
        test_participants = natsorted(folds_df.iloc[self.fold].dropna().tolist())

        train_labels = [1 if p[3] == 'P' else 0 for p in train_participants]
        train_participants, val_participants, _, _ = train_test_split(
            train_participants, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
        train_participants = natsorted(train_participants)
        val_participants = natsorted(val_participants)

        selected_participants = natsorted(train_participants + val_participants + test_participants)
        train_indices = [selected_participants.index(participant) for participant in train_participants]
        val_indices = [selected_participants.index(participant) for participant in val_participants]
        test_indices = [selected_participants.index(participant) for participant in test_participants]

        return participants_df, selected_participants, train_indices, val_indices, test_indices

    def load_raw_data(self, data_type):
        raw_data = list()
        for participant in self.selected_participants:
            participant_path = os.path.join(self.RAW_DATA_PATH, participant)

            if data_type == 'label':
                y = 1 if participant[3] == 'P' else 0
                raw_data.append(y)

            elif data_type == 'text':
                text_segments = [
                    open(os.path.join(participant_path, data), 'r', encoding='utf-8').read()
                    for data in natsorted(os.listdir(participant_path))
                    if data.endswith('.txt')
                ]
                raw_data.append(text_segments)

            elif data_type == 'audio':
                audio_segments = [
                    librosa.load(os.path.join(participant_path, data), sr=self.SAMPLE_RATE)[0]
                    for data in natsorted(os.listdir(participant_path))
                    if data.endswith('.wav')
                ]
                raw_data.append(audio_segments)

        return raw_data
