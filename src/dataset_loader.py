import os
import pandas as pd
from dotenv import load_dotenv
import librosa
from natsort import natsorted
from torch.utils.data import Dataset


class AndroidsCorpusDataset(Dataset):
    def __init__(self, fold: int = 0, train_or_test: str = 'train'):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, os.getenv('DATA_PATH_ANDROIDS_CORPUS'))
        self.fold = fold
        self.train_or_test = train_or_test
        self.SAMPLE_RATE = 16000
        self.X_audio = list()
        self.X_text = list()
        self.y = list()
        self.participants = self.get_participants()
        self.n_participants = len(self.participants)
        self.load_dataset()

    def get_participants(self):
        folds_path = os.path.join(self.DATA_PATH, 'fold-lists.csv')
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
            participant_path = os.path.join(self.DATA_PATH, participant)

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

    def __len__(self):
        return self.n_participants

    def __getitem__(self, idx):
        return self.X_audio[idx], self.X_text[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        X_audio, X_text, y = zip(*batch)
        return list(X_audio), list(X_text), list(y)


class DAICWoZDataset(Dataset):
    def __init__(self, train_or_dev: str = 'train'):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, os.getenv('DATA_PATH_DAIC_WOZ'))
        self.DOWNLOAD_ADDRESS = os.getenv('DOWNLOAD_ADDRESS_DAIC_WOZ')
        self.train_or_dev = train_or_dev
        self.SAMPLE_RATE = 16000
        self.X_audio = list()
        self.X_text = list()
        self.y = list()
        self.participants = self.get_participants()
        self.n_participants = len(self.participants)
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
            text_path = os.path.join(self.DATA_PATH, f'{participant}_TRANSCRIPT.csv')
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

            audio_path = os.path.join(self.DATA_PATH, f'{participant}_AUDIO.wav')
            waveform, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
            interview_df['start_time'] = (interview_df['start_time'] * self.SAMPLE_RATE).astype(int)
            interview_df['stop_time'] = (interview_df['stop_time'] * self.SAMPLE_RATE).astype(int)
            audio_segments = [waveform[segment.start_time:segment.stop_time] for segment in interview_df.itertuples()]
            self.X_audio.append(audio_segments)

    def __len__(self):
        return self.n_participants

    def __getitem__(self, idx):
        return self.X_audio[idx], self.X_text[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        X_audio, X_text, y = zip(*batch)
        return list(X_audio), list(X_text), list(y)
