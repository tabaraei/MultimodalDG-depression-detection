from src.dataset_loader import AndroidsCorpusDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm
import os
import librosa
import re


class Visualization:
    def __init__(self):
        load_dotenv()
        self.PROJECT_ROOT_PATH = os.getenv('PROJECT_ROOT_PATH')
        self.SAMPLE_RATE = 16000
        self.DAIC_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/DAIC_WoZ')
        self.AC_DATA_PATH = os.path.join(self.PROJECT_ROOT_PATH, 'data/raw/Androids_Corpus')

    def waveform_length_histogram(self, dataset):
        durations = list()
        labels = list()

        if dataset == 'DAIC_WoZ':
            for file in tqdm(os.listdir(self.DAIC_DATA_PATH)):
                if file.endswith('.wav'):
                    AUDIO_PATH = os.path.join(self.DAIC_DATA_PATH, file)
                    waveform, sr = librosa.load(AUDIO_PATH, sr=self.SAMPLE_RATE)
                    waveform_length = round(len(waveform) / (self.SAMPLE_RATE * 60))
                    durations.append(waveform_length)

        elif dataset == 'Androids_Corpus':
            pattern = r'^[0-9]{2}_[CP][MF][0-9]{2}_[x0-9]$'
            for folder in tqdm(os.listdir(self.AC_DATA_PATH)):
                if re.match(pattern, folder):
                    FOLDER_PATH = os.path.join(self.AC_DATA_PATH, folder)
                    duration = 0
                    for file in os.listdir(FOLDER_PATH):
                        if file.endswith('.wav'):
                            AUDIO_PATH = os.path.join(FOLDER_PATH, file)
                            waveform, sr = librosa.load(AUDIO_PATH, sr=self.SAMPLE_RATE)
                            duration += len(waveform) / self.SAMPLE_RATE
                    durations.append(duration)
                    labels.append('Depressed' if '_P' in folder else 'Control')

        df = pd.DataFrame({'duration': durations, 'label': labels})
        df = df.sort_values(by='label', ascending=False).reset_index(drop=True)
        fig = plt.figure(figsize=(16, 3))
        ax = sns.barplot(x=df.index, y='duration', hue='label', data=df,
                         palette={'Control': '#f96f00', 'Depressed': '#325097'}, dodge=False)
        ax.set(xlabel='Participant', ylabel='Duration (seconds)')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()
        plt.xticks([])
        plt.tight_layout()
        return fig

    def waveform_sample(self, idx):
        device = torch.device('cpu')
        dataset = AndroidsCorpusDataset(train_val_test='val', device=device)
        x = dataset[0][0][idx][:96000]

        fig = plt.figure(figsize=(10, 3))
        librosa.display.waveshow(x, sr=16000, color='gray', alpha=0.9, linewidth=0.8)
        plt.axis('off')
        plt.tight_layout(pad=0)
        return fig
