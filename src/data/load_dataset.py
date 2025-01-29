import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm.auto import tqdm
import pandas as pd
import audiofile
from dotenv import load_dotenv



class EATD_Corpus:
    def __init__(self, download: bool = False):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.EATD_Corpus_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data', 'raw', 'EATD_Corpus')
        self.EATD_Corpus_DOWNLOAD_PATH = (
            'https://3f7xrg.bl.files.1drv.com/y4mzCOdmCDMRHErLHsESWkD0rmmY1j9ca3CfhfCpv6poE3j'
            '-0dZd9HmKVC3k0LWif3I2XgyC1tErV8SrVr1mJNVNHYPmU_qqNvvZVBhOijBfsdwWaYVs6Zd4QzsC4HaljGNbTWwtnQ'
            '-JrWog9EB0DbblDlKlNBYxcroYpLW9_qrHX7Ub2XEnYVcZ1gqMptzr3Us9Jj66IdrWRLoaYK_FJWiRQ'
        )
        self.EATD_Corpus_DOWNLOAD_PASSWORD = 'Ymj26Uv5'
        self.SAMPLE_RATE = None
        if download: self.download_dataset()

    def download_dataset(self):
        os.makedirs(self.EATD_Corpus_DATA_PATH, exist_ok=True)
        response = requests.get(self.EATD_Corpus_DOWNLOAD_PATH, stream=True)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(self.EATD_Corpus_DATA_PATH, pwd=bytes(self.EATD_Corpus_DOWNLOAD_PASSWORD, 'utf-8'))

        DOWNLOAD_PATH = os.path.join(self.EATD_Corpus_DATA_PATH, 'EATD-Corpus')
        for folder in os.listdir(DOWNLOAD_PATH):
            folder_path = os.path.join(DOWNLOAD_PATH, folder)
            shutil.move(folder_path, self.EATD_Corpus_DATA_PATH)
        os.rmdir(DOWNLOAD_PATH)


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
