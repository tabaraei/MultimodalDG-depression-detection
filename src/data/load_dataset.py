import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm.auto import tqdm
import pandas as pd
import audiofile
from dotenv import load_dotenv
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import re


class Androids_Corpus:
    def __init__(self, download: bool = False, extract_transcripts: bool = False):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.Androids_Corpus_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data', 'raw', 'Androids_Corpus')
        self.Androids_Corpus_DOWNLOAD_PATH = 'https://www.dropbox.com/scl/fi/74bu3kf0pbmo4x4zntdk7/Androids-Corpus.zip?rlkey=0sl5ktwq8lx99a4bsux8xdsl3&e=2&dl=1'
        self.SAMPLE_RATE = 16000
        if download: self.download_dataset()
        if extract_transcripts: self.extract_transcripts()
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

    def extract_transcripts(self):
        # Prepare and download the Whisper fine-tuned model
        device = (
            'mps:0' if torch.backends.mps.is_available() else
            'cuda:0' if torch.cuda.is_available() else
            'cpu'
        )
        torch_dtype = torch.float32 if device == 'cpu' else torch.float16
        MODEL_NAME = 'bofenghuang/whisper-large-v3-distil-it-v0.2'
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        pipe = pipeline(
            task='automatic-speech-recognition',
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={
                'task': 'transcribe',
                'language': 'it',
                'return_timestamps': True,
                'max_new_tokens': 128,
                'forced_decoder_ids': None,
            }
        )

        # Extract transcripts
        pattern = r'^[0-9]{2}_[CP][MF][0-9]{2}_[x0-9]$'
        PARTICIPANTS_PATH = os.listdir(self.Androids_Corpus_DATA_PATH)
        PARTICIPANTS_PATH = [f for f in PARTICIPANTS_PATH if re.match(pattern, f)]
        for participant in tqdm(sorted(PARTICIPANTS_PATH)):
            PARTICIPANT_PATH = os.path.join(self.Androids_Corpus_DATA_PATH, participant)

            for data in sorted(os.listdir(PARTICIPANT_PATH)):
                DATA_PATH = os.path.join(PARTICIPANT_PATH, data)
                if DATA_PATH.endswith('.wav'):
                    waveform, _ = librosa.load(DATA_PATH, sr=self.SAMPLE_RATE)
                    text = pipe(waveform)['text']

                    TEXT_FILE_NAME = DATA_PATH.replace('.wav', '.txt')
                    with open(TEXT_FILE_NAME, 'w') as f:
                        f.write(text)

    def extract_fold_segments(self):
        FOLDS_PATH = os.path.join(self.Androids_Corpus_DATA_PATH, 'fold-lists.csv')
        df = pd.read_csv(FOLDS_PATH)

        interview_columns = df.columns[df.columns.tolist().index('Interview'):]
        folds = df.loc[1:, interview_columns].T.apply(lambda x: [
            {'Participant_ID': val[1:-1], 'Depressed': 1 if val[4] == 'P' else 0}
            for val in x.dropna().tolist()], axis=1).tolist()

        for participants_data in folds:
            for participant in participants_data:
                PARTICIPANT_PATH = os.path.join(self.Androids_Corpus_DATA_PATH, participant['Participant_ID'])
                audio_segments, text_segments = list(), list()

                for data in sorted(os.listdir(PARTICIPANT_PATH)):
                    DATA_PATH = os.path.join(PARTICIPANT_PATH, data)
                    if data.endswith('.wav'):
                        waveform, _ = librosa.load(DATA_PATH, sr=self.SAMPLE_RATE)
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
            ZIP_PATH = os.path.join(self.DAIC_WoZ_DOWNLOAD_PATH, zip_name)
            response = requests.get(ZIP_PATH, stream=True)
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
            TEXT_PATH = os.path.join(self.DAIC_WoZ_DATA_PATH, f"{participant['Participant_ID']}_TRANSCRIPT.csv")
            AUDIO_PATH = os.path.join(self.DAIC_WoZ_DATA_PATH, f"{participant['Participant_ID']}_AUDIO.wav")

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
            waveform, _ = librosa.load(AUDIO_PATH, sr=self.SAMPLE_RATE)
            interview_df['start_time'] = (interview_df['start_time'] * self.SAMPLE_RATE).astype(int)
            interview_df['stop_time'] = (interview_df['stop_time'] * self.SAMPLE_RATE).astype(int)
            audio_segments = [waveform[segment.start_time:segment.stop_time] for segment in interview_df.itertuples()]

            # Assign the extracted segments to the participant
            participant['Text_Segments'] = text_segments
            participant['Audio_Segments'] = audio_segments

        return participants_data
