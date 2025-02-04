import torch
import torch.nn as nn


class MultimodalClassifier(nn.Module):
    """
    1. First, audio and text segments are fed through the vectorizers, taking as output a sequence of vectors with
       dimensionality <batch_size, seq_len, feature_dim>, where `seq_len` depends on the audio and text lengths.

    2. Then, we feed the sequence of vectors through the BiLSTM layers, one for each modality, resulting in output
       of dimensionality <num_layers * 2 (BiLSTM), batch_size, hidden_dim>. Then, we take the last forward/backward
       hidden state layers as output, resulting in <batch_size, hidden_dim * 2 (BiLSTM)>

    3. Finally, we concatenate the extracted features from the BiLSTM layers for audio and text modalities, giving
       output dimensionality of <batch_size, hidden_dim * 2 (BiLSTM) * 2 (modalities)>, and then feed them through
       a fully connected network with `fc_hidden_dim` hidden neurons and a single output neuron.
    """

    def __init__(self, audio_vectorizer, text_vectorizer, hidden_dim=256, num_layers=1, fc_hidden_dim=128):
        super().__init__()
        self.text_vectorizer = text_vectorizer
        self.audio_vectorizer = audio_vectorizer
        self.text_lstm = nn.LSTM(
            input_size=self.text_vectorizer.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.audio_lstm = nn.LSTM(
            input_size=self.audio_vectorizer.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, fc_hidden_dim),  # <hidden_dim * forward/backward * 2 modalities>
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_segments, text_segments):
        # output_dim: <batch_size, seq_len, feature_dim>
        audio_features = self.audio_vectorizer(audio_segments)
        text_features = self.text_vectorizer(text_segments)

        # output_dim: <num_layers * 2 (BiLSTM), batch_size, hidden_dim>
        _, (BiLSTM_audio_hidden, _) = self.audio_lstm(audio_features)
        _, (BiLSTM_text_hidden, _) = self.text_lstm(text_features)

        # output_dim: <batch_size, hidden_dim * 2 (BiLSTM)>
        BiLSTM_audio_hidden = torch.cat((BiLSTM_audio_hidden[-2], BiLSTM_audio_hidden[-1]), dim=-1)
        BiLSTM_text_hidden = torch.cat((BiLSTM_text_hidden[-2], BiLSTM_text_hidden[-1]), dim=-1)

        # output_dim: <batch_size, hidden_dim * 2 (BiLSTM) * 2 (modalities)>
        concatenated_features = torch.cat((BiLSTM_audio_hidden, BiLSTM_text_hidden), dim=-1)

        # output_dim: <batch_size, 1>
        output = self.fc(concatenated_features)
        return output

    def compute_loss(self, predictions, targets):
        loss_fn = nn.BCELoss()
        return loss_fn(predictions, targets)
