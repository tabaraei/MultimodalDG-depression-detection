import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm.auto import tqdm
import pandas as pd
import audiofile
from dotenv import load_dotenv


class Androids_Corpus:
    def __init__(self, download: bool = False):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.Androids_Corpus_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data', 'raw', 'Androids_Corpus')
        self.Androids_Corpus_DOWNLOAD_PATH = 'https://www.dropbox.com/scl/fi/74bu3kf0pbmo4x4zntdk7/Androids-Corpus.zip?rlkey=0sl5ktwq8lx99a4bsux8xdsl3&e=2&dl=1'
        self.SAMPLE_RATE = 44100
        if download: self.download_dataset()
        self.fold_segments = self.extract_fold_segments()

    def download_dataset(self):
        os.makedirs(self.Androids_Corpus_DATA_PATH, exist_ok=True)
        response = requests.get(self.Androids_Corpus_DOWNLOAD_PATH, stream=True)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(self.Androids_Corpus_DATA_PATH)

        FOLDS_PATH = os.path.join(self.Androids_Corpus_DATA_PATH, 'Androids-Corpus/fold-lists.csv')
        shutil.move(FOLDS_PATH, self.Androids_Corpus_DATA_PATH)

        AUDIO_PATH = os.path.join(self.Androids_Corpus_DATA_PATH, 'Androids-Corpus/Interview-Task/audio_clip')
        for content in os.listdir(AUDIO_PATH):
            shutil.move(os.path.join(AUDIO_PATH, content), self.Androids_Corpus_DATA_PATH)

        DOWNLOAD_PATH = os.path.join(self.Androids_Corpus_DATA_PATH, 'Androids-Corpus')
        shutil.rmtree(DOWNLOAD_PATH)

        macosx_dir = os.path.join(self.Androids_Corpus_DATA_PATH, '__MACOSX')
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)

    def extract_fold_segments(self):
        FOLDS_PATH = os.path.join(self.Androids_Corpus_DATA_PATH, 'fold-lists.csv')
        df = pd.read_csv(FOLDS_PATH)

        interview_columns = df.columns[df.columns.tolist().index('Interview'):]
        folds = df.loc[1:, interview_columns].T.apply(lambda x: [
            {'Participant_ID': val[1:-1], 'Depressed': 1 if val[4] == 'P' else 0}
            for val in x.dropna().tolist()], axis=1).tolist()

        for participants_data in folds:
            for participant in participants_data:
                PARTICIPANT_PATH = f"{self.Androids_Corpus_DATA_PATH}/{participant['Participant_ID']}"
                audio_segments, text_segments = list(), list()

                for data in sorted(os.listdir(PARTICIPANT_PATH)):
                    DATA_PATH = os.path.join(PARTICIPANT_PATH, data)
                    if data.endswith('.wav'):
                        waveform, sample_rate = audiofile.read(DATA_PATH, dtype='float32')
                        audio_segments.append(waveform)
                    else:
                        with open(DATA_PATH, 'r', encoding='utf-8') as f:
                            text_segments.append(f.read())

                participant['Text_Segments'] = text_segments
                participant['Audio_Segments'] = audio_segments

        return folds


class DAIC_WoZ:
    def __init__(self, download: bool = False):
        """
        [DAIC-WoZ]
        Excluded sessions: 342,394,398,460
        Included sessions with special notes:
            - 373: there is an interruption around 5:52-7:00
            - 444: there is an interruption around 4:46-6:27
            - 451,458,480: sessions are technically complete, but missing Ellie
            - 402: video recording is cut ~2min before the end

        [AMHD-GPT]
        When evaluating the data, the following files are not included in the training,
        because they are described in the documentation of the dataset as noisy or
        interrupted transcriptions: 373 and 444. The files where Ellie is missing
        (451, 458 and 480) are not removed because only the statements of the participants
        are considered.
        """
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.DAIC_WoZ_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data', 'raw', 'DAIC_WoZ')
        self.DAIC_WoZ_DOWNLOAD_PATH = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'
        self.SAMPLE_RATE = 16000
        self.excluded_sessions = {342, 394, 398, 460} | {373, 444}
        self.selected_sessions = set(range(300, 493)) - self.excluded_sessions
        if download: self.download_dataset()
        self.train_data = self.extract_segments(train_or_dev='train')
        self.test_data = self.extract_segments(train_or_dev='dev')

    def download_dataset(self):
        """
        Directly downloads the audio and text files from the DAIC-WoZ website
        Saves the data in the dedicated destination at "self.DAIC_WoZ_DATA_PATH"
        """
        os.makedirs(self.DAIC_WoZ_DATA_PATH, exist_ok=True)
        zip_files = [f'{session}_P.zip' for session in self.selected_sessions]
        for zip_name in tqdm(zip_files):
            response = requests.get(f'{self.DAIC_WoZ_DOWNLOAD_PATH}/{zip_name}', stream=True)
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as zf:
                prefix = zip_name[:3]
                zf.extract(f'{prefix}_TRANSCRIPT.csv', path=self.DAIC_WoZ_DATA_PATH)
                zf.extract(f'{prefix}_AUDIO.wav', path=self.DAIC_WoZ_DATA_PATH)

    def extract_segments(self, train_or_dev: str = 'train'):
        """
        [AMHD-GPT]
        The score after filling out the PHQ-8 questionnaire from file 409 is 10. This was
        wrongly listed as not depressive. For this reason, this label is corrected manually.
        """
        participants_df = pd.read_csv(f'{self.DAIC_WoZ_DOWNLOAD_PATH}/{train_or_dev}_split_Depression_AVEC2017.csv')
        participants_df = participants_df[~participants_df['Participant_ID'].isin(self.excluded_sessions)]

        # Set the depression label manually (since error exists)
        score_column = 'PHQ8_Score' if 'PHQ8_Score' in participants_df.columns else 'PHQ_Score'
        participants_df['Depressed'] = (participants_df[score_column] >= 10).astype(int)

        # Create a data structure, with each entry specific to a single participant
        participants_data = participants_df[['Participant_ID', 'Depressed']].to_dict(orient='records')
        for participant in participants_data:
            # Set the path for audio and text file of each participant
            TEXT_PATH = f"{self.DAIC_WoZ_DATA_PATH}/{participant['Participant_ID']}_TRANSCRIPT.csv"
            AUDIO_PATH = f"{self.DAIC_WoZ_DATA_PATH}/{participant['Participant_ID']}_AUDIO.wav"

            # Load the interview text, aggregate consecutive rows of participant, and extract text segments
            interview_df = pd.read_csv(TEXT_PATH, delimiter='\t')
            interview_df['group'] = (interview_df['speaker'] != interview_df['speaker'].shift()).cumsum()
            interview_df = interview_df.dropna().groupby('group').agg({
                'start_time': 'min',
                'stop_time': 'max',
                'speaker': 'first',
                'value': '. '.join
            }).reset_index(drop=True)
            interview_df = interview_df[interview_df['speaker'] == 'Participant'].copy()
            text_segments = interview_df['value'].tolist()

            # Load the interview audio, set the start/end of segments based on sample rate, extract audio segment
            waveform, sample_rate = audiofile.read(AUDIO_PATH, dtype='float32')
            interview_df['start_time'] = (interview_df['start_time'] * sample_rate).astype(int)
            interview_df['stop_time'] = (interview_df['stop_time'] * sample_rate).astype(int)
            audio_segments = [waveform[segment.start_time:segment.stop_time] for segment in interview_df.itertuples()]

            # Assign the extracted segments to the participant
            participant['Text_Segments'] = text_segments
            participant['Audio_Segments'] = audio_segments

        return participants_data
