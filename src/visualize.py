from src.dataset_loader import DAICWoZDataset, AndroidsCorpusDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE


class Visualization:
    def t_SNE_distributions(self, audio_vectorizer, text_vectorizer, train_or_test='train', fold=0):
        DAIC_dataset = DAICWoZDataset(
            train_or_test=train_or_test,
            audio_vectorizer=audio_vectorizer,
            text_vectorizer=text_vectorizer
        )
        AC_dataset = AndroidsCorpusDataset(
            fold=fold,
            train_or_test=train_or_test,
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

        plt.figure(figsize=(8, 4))
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
        plt.show()
