import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm.auto import tqdm
from dotenv import load_dotenv
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import re
from natsort import natsorted


class DatasetDownloader:
    """
    1. Directly downloads the audio and text files from the corresponding website
    2. Saves the data in the dedicated destination at 'self.DATA_PATH'
    3. Extracts the transcripts for the audio files in case of Androids-Corpus dataset
    """

    def __init__(self, dataset):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.SAMPLE_RATE = 16000

        if dataset == 'DAIC_WoZ':
            self.RAW_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/DAIC_WoZ')
            self.DOWNLOAD_ADDRESS = os.getenv('DOWNLOAD_ADDRESS_DAIC_WOZ')
            self.download_DAIC_WoZ()
        elif dataset == 'Androids_Corpus':
            self.RAW_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/Androids_Corpus')
            self.DOWNLOAD_ADDRESS = os.getenv('DOWNLOAD_ADDRESS_ANDROIDS_CORPUS')
            self.download_Androids_Corpus()
            self.extract_transcripts()

    def download_DAIC_WoZ(self):
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

        excluded_sessions = {342, 394, 398, 460} | {373, 444}
        selected_sessions = set(range(300, 493)) - excluded_sessions

        for session in selected_sessions:
            ZIP_PATH = os.path.join(self.DOWNLOAD_ADDRESS, f'{session}_P.zip')
            response = requests.get(ZIP_PATH, stream=True)
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as zf:
                zf.extract(f'{session}_P_TRANSCRIPT.csv', path=self.RAW_DATA_PATH)
                zf.extract(f'{session}_P_AUDIO.wav', path=self.RAW_DATA_PATH)

    def download_Androids_Corpus(self):
        os.makedirs(self.RAW_DATA_PATH, exist_ok=True)
        response = requests.get(self.DOWNLOAD_ADDRESS, stream=True)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(self.RAW_DATA_PATH)

        FOLDS_PATH = os.path.join(self.RAW_DATA_PATH, 'Androids-Corpus/fold-lists.csv')
        shutil.move(FOLDS_PATH, self.RAW_DATA_PATH)

        AUDIO_PATH = os.path.join(self.RAW_DATA_PATH, 'Androids-Corpus/Interview-Task/audio_clip')
        for content in os.listdir(AUDIO_PATH):
            shutil.move(os.path.join(AUDIO_PATH, content), self.RAW_DATA_PATH)

        DOWNLOAD_PATH = os.path.join(self.RAW_DATA_PATH, 'Androids-Corpus')
        shutil.rmtree(DOWNLOAD_PATH)

    def extract_transcripts(self):
        # Prepare and download the Whisper fine-tuned model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODEL_NAME = 'bofenghuang/whisper-large-v3-distil-it-v0.2'
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to(device)
        pipe = pipeline(
            task='automatic-speech-recognition',
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            device=device,
            generate_kwargs={'task': 'transcribe', 'language': 'it'}
        )

        # Extract transcripts
        pattern = r'^[0-9]{2}_[CP][MF][0-9]{2}_[x0-9]$'
        for participant in tqdm(natsorted(os.listdir(self.RAW_DATA_PATH))):
            if re.match(pattern, participant):
                PARTICIPANT_PATH = os.path.join(self.RAW_DATA_PATH, participant)

                for data in natsorted(os.listdir(PARTICIPANT_PATH)):
                    data_path = os.path.join(PARTICIPANT_PATH, data)
                    if data_path.endswith('.wav'):
                        waveform, _ = librosa.load(data_path, sr=self.SAMPLE_RATE)
                        text = pipe(waveform)['text']

                        text_file_name = data_path.replace('.wav', '.txt')
                        with open(text_file_name, 'w') as f:
                            f.write(text)
