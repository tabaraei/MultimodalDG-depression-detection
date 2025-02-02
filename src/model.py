import torch
import torch.nn as nn


class MultimodalBiLSTMClassifier(nn.Module):
    def __init__(self, text_vectorizer, audio_vectorizer, hidden_dim=256, num_layers=2, fc_hidden_dim=128):
        super().__init__()

        self.text_vectorizer = text_vectorizer  # Any transformer-based text vectorizer (e.g., BERT, RoBERTa)
        self.audio_vectorizer = audio_vectorizer  # Any audio feature extractor (e.g., wav2vec2, melspectrogram)

        # Determine feature sizes based on vectorizers
        text_feat_dim = self.text_vectorizer.feature_dim
        audio_feat_dim = self.audio_vectorizer.feature_dim

        # BiLSTM for text
        self.text_lstm = nn.LSTM(input_size=text_feat_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=True)

        # BiLSTM for audio
        self.audio_lstm = nn.LSTM(input_size=audio_feat_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                  batch_first=True, bidirectional=True)

        # Fully connected network
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim * 2, fc_hidden_dim),  # 2 for BiLSTM (forward + backward) * 2 modalities
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_input, audio_input):
        # Vectorize text and audio
        text_features = self.text_vectorizer(text_input)  # Shape: (batch, seq_len, text_feat_dim)
        audio_features = self.audio_vectorizer(audio_input)  # Shape: (batch, seq_len, audio_feat_dim)

        # BiLSTM processing (taking the last hidden states from both directions)
        _, (text_hidden, _) = self.text_lstm(text_features)  # text_hidden: (num_layers * 2, batch, hidden_dim)
        _, (audio_hidden, _) = self.audio_lstm(audio_features)  # audio_hidden: (num_layers * 2, batch, hidden_dim)

        # Concatenating last forward and backward hidden states
        text_out = torch.cat((text_hidden[-2], text_hidden[-1]), dim=-1)  # Shape: (batch, hidden_dim * 2)
        audio_out = torch.cat((audio_hidden[-2], audio_hidden[-1]), dim=-1)  # Shape: (batch, hidden_dim * 2)

        # Concatenating text and audio outputs
        combined_features = torch.cat((text_out, audio_out), dim=-1)  # Shape: (batch, hidden_dim * 4)

        # Fully connected network for classification
        output = self.fc(combined_features)  # Shape: (batch, 1)
        return output

    def compute_loss(self, predictions, targets):
        loss_fn = nn.BCELoss()
        return loss_fn(predictions, targets)
