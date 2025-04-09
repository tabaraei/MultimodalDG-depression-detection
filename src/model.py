import torch
import torch.nn as nn
from block import fusions


class MultimodalClassifier(nn.Module):
    """
    1. First, audio and text segments are fed through the vectorizers, taking as output a sequence of vectors with
       dimensionality <batch_size, seq_len, feature_dim>, where `seq_len` depends on the audio and text lengths.

    2. Then, we feed the sequence of vectors through the BiLSTM layers, one for each modality, resulting in output
       of dimensionality <lstm_n_layers * 2 (BiLSTM), batch_size, lstm_hidden_dim>. Then, we take the last forward/backward
       hidden state layers as output, resulting in <batch_size, lstm_hidden_dim * 2 (BiLSTM)>

    3. Finally, we concatenate the extracted features from the BiLSTM layers for audio and text modalities, giving
       output dimensionality of <batch_size, lstm_hidden_dim * 2 (BiLSTM) * 2 (modalities)>, and then feed them through
       a fully connected network with `fc_hidden_dim` hidden neurons and a single output neuron.
    """

    def __init__(
            self,
            modality,
            audio_feature_dim,
            text_feature_dim,
            audio_lstm_hidden_dim,
            text_lstm_hidden_dim,
            fc_hidden_dim,
            lstm_n_layers=1
    ):
        super().__init__()
        self.modality = modality
        self.audio_feature_dim = audio_feature_dim
        self.text_feature_dim = text_feature_dim
        if modality == 'audio':
            self.linear_input_dim = audio_lstm_hidden_dim
        elif modality == 'text':
            self.linear_input_dim = text_lstm_hidden_dim
        elif modality == 'multimodal':
            self.linear_input_dim = audio_lstm_hidden_dim + text_lstm_hidden_dim

        self.audio_lstm = nn.LSTM(
            input_size=self.audio_feature_dim,
            hidden_size=audio_lstm_hidden_dim,
            num_layers=lstm_n_layers,
            batch_first=True,
            bidirectional=True
        )
        self.text_lstm = nn.LSTM(
            input_size=self.text_feature_dim,
            hidden_size=text_lstm_hidden_dim,
            num_layers=lstm_n_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.linear_input_dim * 2, fc_hidden_dim),  # forward/backward
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )

    def extract_BiLSTM_hidden(self, lstm, x):
        # x_dim: <batch_size, seq_len, feature_dim>
        # output_dim: <lstm_n_layers * 2 (BiLSTM), batch_size, lstm_hidden_dim>
        _, (BiLSTM_hidden, _) = lstm(x)
        # output_dim: <batch_size, lstm_hidden_dim * 2 (BiLSTM)>
        return torch.cat((BiLSTM_hidden[-2], BiLSTM_hidden[-1]), dim=-1)

    def forward(self, x_audio, x_text):
        if self.modality == 'audio':
            features = self.extract_BiLSTM_hidden(self.audio_lstm, x_audio)
        elif self.modality == 'text':
            features = self.extract_BiLSTM_hidden(self.text_lstm, x_text)
        elif self.modality == 'multimodal':
            audio_features = self.extract_BiLSTM_hidden(self.audio_lstm, x_audio)
            text_features = self.extract_BiLSTM_hidden(self.text_lstm, x_text)
            # output_dim: <batch_size, lstm_hidden_dim * 2 (BiLSTM) * 2 (modalities)>
            features = torch.cat((audio_features, text_features), dim=-1)

        # output_dim: <batch_size, 1>
        output_logits = self.fc(features)
        return torch.mean(output_logits).unsqueeze(0)
