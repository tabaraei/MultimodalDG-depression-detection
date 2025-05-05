from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
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

    @staticmethod
    def t_SNE_distributions(audio_vectorizer, text_vectorizer, train_val_test='train', fold=0):
        DAIC_dataset = DAICWoZDataset(
            train_val_test=train_val_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer
        )
        AC_dataset = AndroidsCorpusDataset(
            fold=fold,
            train_val_test=train_val_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer
        )
        get_avg_features = lambda X_audio, X_text: np.array(
            [torch.cat([x1.squeeze().mean(axis=0), x2.squeeze().mean(axis=0)], dim=0)
             for x1, x2 in zip(X_audio, X_text)]
        )

        X_DAIC = get_avg_features(DAIC_dataset.X_audio, DAIC_dataset.X_text)
        X_AC = get_avg_features(AC_dataset.X_audio, AC_dataset.X_text)
        X_combined = np.vstack([X_DAIC, X_AC])
        X_combined = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_combined)
        X_DAIC = X_combined[:len(X_DAIC), :]
        X_AC = X_combined[len(X_DAIC):, :]

        y_DAIC = np.array(DAIC_dataset.y).squeeze().astype(int)
        y_AC = np.array(AC_dataset.y).squeeze().astype(int)

        df_DAIC = pd.DataFrame({'TSNE_1': X_DAIC[:, 0], 'TSNE_2': X_DAIC[:, 1], 'Label': y_DAIC, 'Dataset': 'DAIC_WoZ'})
        df_AC = pd.DataFrame({'TSNE_1': X_AC[:, 0], 'TSNE_2': X_AC[:, 1], 'Label': y_AC, 'Dataset': 'Androids_Corpus'})

        fig = plt.figure(figsize=(8, 4))
        common_args = {
            'x': 'TSNE_1',
            'y': 'TSNE_2',
            'hue': 'Label',
            'palette': {0: 'red', 1: 'blue'},
            'alpha': 0.7,
            's': 100
        }
        new_legends = [
            'DAIC-WoZ (Not Depressed)',
            'DAIC-WoZ (Depressed)',
            'Androids-Corpus (Not Depressed)',
            'Androids-Corpus (Depressed)',
        ]

        sns.scatterplot(data=df_DAIC, marker='o', **common_args)
        sns.scatterplot(data=df_AC, marker='x', **common_args)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Feature Distributions')
        legend = plt.legend()
        for i, text in enumerate(legend.texts):
            text.set_text(new_legends[i])
        return fig
