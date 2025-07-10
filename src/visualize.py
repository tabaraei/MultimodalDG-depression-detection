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

    def results(self):
        cols = ["Audio Vectorizer", "Text Vectorizer", "Accuracy", "Precision", "Recall", "F1-Score"]

        df_20s = pd.DataFrame([
            ["MelSpec", "BERT", 77.18, 76.23, 88.10, 80.47],
            ["MelSpec", "ItalianBERT", 86.37, 86.76, 89.49, 87.34],
            ["MelSpec", "XLM-RoBERTa", 68.45, 66.05, 92.65, 76.07],
            ["HuBERT", "BERT", 76.21, 74.07, 86.39, 78.80],
            ["HuBERT", "ItalianBERT", 83.06, 80.44, 91.71, 85.03],
            ["HuBERT", "XLM-RoBERTa", 70.44, 71.36, 81.61, 72.81],
            ["Wav2Vec2", "BERT", 78.79, 77.63, 86.59, 80.93],
            ["Wav2Vec2", "ItalianBERT", 86.81, 85.36, 93.46, 88.25],
            ["Wav2Vec2", "XLM-RoBERTa", 73.27, 73.31, 79.63, 74.99]
        ], columns=cols)

        df_30s = pd.DataFrame([
            ["MelSpec", "BERT", 85.19, 85.91, 89.29, 86.83],
            ["MelSpec", "ItalianBERT", 90.37, 90.77, 92.00, 90.77],
            ["MelSpec", "XLM-RoBERTa", 67.88, 69.07, 85.37, 74.21],
            ["HuBERT", "BERT", 77.88, 76.88, 87.97, 80.92],
            ["HuBERT", "ItalianBERT", 82.49, 80.58, 91.08, 84.74],
            ["HuBERT", "XLM-RoBERTa", 71.27, 72.43, 87.67, 75.97],
            ["Wav2Vec2", "BERT", 81.02, 79.60, 90.44, 83.75],
            ["Wav2Vec2", "ItalianBERT", 86.22, 85.90, 92.16, 88.09],
            ["Wav2Vec2", "XLM-RoBERTa", 73.87, 75.03, 82.73, 76.58]
        ], columns=cols)

        df_45s = pd.DataFrame([
            ["MelSpec", "BERT", 84.31, 84.85, 87.29, 85.42],
            ["MelSpec", "ItalianBERT", 88.10, 87.09, 90.89, 88.72],
            ["MelSpec", "XLM-RoBERTa", 70.77, 69.81, 85.61, 74.97],
            ["HuBERT", "BERT", 77.29, 77.51, 87.38, 80.43],
            ["HuBERT", "ItalianBERT", 85.36, 85.72, 89.49, 86.63],
            ["HuBERT", "XLM-RoBERTa", 70.75, 71.67, 83.18, 74.74],
            ["Wav2Vec2", "BERT", 79.67, 84.54, 82.75, 80.36],
            ["Wav2Vec2", "ItalianBERT", 83.07, 85.49, 87.73, 85.08],
            ["Wav2Vec2", "XLM-RoBERTa", 70.95, 71.53, 81.79, 74.31]
        ], columns=cols)

        df_60s = pd.DataFrame([
            ["MelSpec", "BERT", 85.79, 86.97, 88.89, 87.13],
            ["MelSpec", "ItalianBERT", 87.47, 90.43, 87.82, 88.01],
            ["MelSpec", "XLM-RoBERTa", 75.45, 75.00, 86.17, 78.89],
            ["HuBERT", "BERT", 79.91, 81.06, 85.42, 82.02],
            ["HuBERT", "ItalianBERT", 80.19, 80.80, 87.06, 82.45],
            ["HuBERT", "XLM-RoBERTa", 76.42, 75.50, 86.53, 78.59],
            ["Wav2Vec2", "BERT", 80.83, 83.55, 83.01, 82.05],
            ["Wav2Vec2", "ItalianBERT", 83.97, 85.05, 87.31, 85.46],
            ["Wav2Vec2", "XLM-RoBERTa", 81.57, 81.06, 86.05, 82.70]
        ], columns=cols)

        # segment durations' plot
        x1 = df_20s.iloc[:, 2:].mean()
        x2 = df_30s.iloc[:, 2:].mean()
        x3 = df_45s.iloc[:, 2:].mean()
        x4 = df_60s.iloc[:, 2:].mean()
        df_plot = pd.DataFrame({
            '20s': x1,
            '30s': x2,
            '45s': x3,
            '60s': x4
        }).reset_index().melt(id_vars='index', var_name='segment_duration', value_name='Score')
        df_plot = df_plot.rename(columns={'index': 'Metric'})
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=df_plot, x='Metric', y='Score', hue='segment_duration', marker='o', palette='Set2', ax=ax1)
        ax1.set_ylabel("Percentage (%)")
        ax1.set_xlabel("")
        ax1.set_ylim(bottom=70)
        ax1.legend(loc='lower right', title="segment_duration")
        ax1.grid(True)
        fig1.tight_layout()

        # feature extractors' plot
        all_data = pd.concat([df_20s, df_30s, df_45s, df_60s], ignore_index=True)
        avg_metrics = all_data.groupby(["Audio Vectorizer", "Text Vectorizer"]).mean(numeric_only=True).reset_index()
        avg_metrics['Combo'] = avg_metrics['Audio Vectorizer'] + ' / ' + avg_metrics['Text Vectorizer']
        df_plot = avg_metrics.melt(id_vars='Combo', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                   var_name='Metric', value_name='Score')
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_plot, x='Metric', y='Score', hue='Combo', marker='o', palette='Set3', ax=ax2)
        ax2.set_ylabel("Percentage (%)")
        ax2.set_xlabel("")
        ax2.set_ylim(60, 95)
        ax2.legend(title='Feature Extractor', bbox_to_anchor=(1.02, 1.023))
        ax2.grid(True)
        fig2.tight_layout()

        # return both plots
        return fig1, fig2
