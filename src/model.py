import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class IntraModalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        attn_scores = self.attn(x)
        weights = F.softmax(attn_scores, dim=1)
        weights = self.dropout(weights)
        return (weights * x).sum(dim=1)


class CrossModalAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, context):
        Q = self.query_proj(query)
        K = self.key_proj(context)
        V = self.value_proj(context)

        attn_output, _ = self.attn(Q, K, V)
        output = self.norm(Q + self.dropout(attn_output))
        return output


class FeatureExtractor(nn.Module):
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
            attn_hidden_dim,
            cross_attn_hidden_dim,
            lstm_n_layers=1
    ):
        super().__init__()
        self.modality = modality
        self.audio_feature_dim = audio_feature_dim
        self.text_feature_dim = text_feature_dim
        self.audio_bilstm_dim = audio_lstm_hidden_dim * 2
        self.text_bilstm_dim = text_lstm_hidden_dim * 2

        if modality == 'audio':
            self.output_dim = self.audio_bilstm_dim
        elif modality == 'text':
            self.output_dim = self.text_bilstm_dim
        elif modality == 'multimodal':
            self.output_dim = self.audio_bilstm_dim + self.text_bilstm_dim + cross_attn_hidden_dim

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
        self.audio_norm = nn.LayerNorm(audio_lstm_hidden_dim * 2)
        self.text_norm = nn.LayerNorm(text_lstm_hidden_dim * 2)
        self.audio_attn = IntraModalAttention(input_dim=self.audio_bilstm_dim, hidden_dim=attn_hidden_dim)
        self.text_attn = IntraModalAttention(input_dim=self.text_bilstm_dim, hidden_dim=attn_hidden_dim)
        self.cross_attn_pool = IntraModalAttention(input_dim=cross_attn_hidden_dim, hidden_dim=attn_hidden_dim)
        self.cross_attn = CrossModalAttention(
            query_dim=self.text_bilstm_dim,
            key_dim=self.audio_bilstm_dim,
            hidden_dim=cross_attn_hidden_dim
        )

    def extract_self_attended_BiLSTM(self, x, modality):
        if modality == 'audio':
            BiLSTM_seq, _ = self.audio_lstm(x)
            BiLSTM_seq = self.audio_norm(BiLSTM_seq)
            BiLSTM_attn = self.audio_attn(BiLSTM_seq)
        else:
            BiLSTM_seq, _ = self.text_lstm(x)
            BiLSTM_seq = self.text_norm(BiLSTM_seq)
            BiLSTM_attn = self.text_attn(BiLSTM_seq)
        return BiLSTM_attn, BiLSTM_seq

    def forward(self, x_audio, x_text):
        if self.modality == 'audio':
            features, _ = self.extract_self_attended_BiLSTM(x=x_audio, modality='audio')
        elif self.modality == 'text':
            features, _ = self.extract_self_attended_BiLSTM(x=x_text, modality='text')
        elif self.modality == 'multimodal':
            audio_attn, audio_seq = self.extract_self_attended_BiLSTM(x=x_audio, modality='audio')
            text_attn, text_seq = self.extract_self_attended_BiLSTM(x=x_text, modality='text')
            cross_modal_features = self.cross_attn_pool(self.cross_attn(query=text_seq, context=audio_seq))
            features = torch.cat([audio_attn, text_attn, cross_modal_features], dim=-1)
        return features


class DepressionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )

    def forward(self, features):
        return self.fc(features)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_domains):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=n_domains)
        )

    def forward(self, features):
        return self.discriminator(features)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class Model(nn.Module):
    def __init__(
            self,
            generalization,
            modality,
            audio_feature_dim,
            text_feature_dim,
            audio_lstm_hidden_dim,
            text_lstm_hidden_dim,
            attn_hidden_dim,
            cross_attn_hidden_dim,
            fc_hidden_dim,
            n_domains,
            lambda_grl
    ):
        super().__init__()
        self.generalization = generalization
        self.feature_extractor = FeatureExtractor(
            modality=modality,
            audio_feature_dim=audio_feature_dim,
            text_feature_dim=text_feature_dim,
            audio_lstm_hidden_dim=audio_lstm_hidden_dim,
            text_lstm_hidden_dim=text_lstm_hidden_dim,
            attn_hidden_dim=attn_hidden_dim,
            cross_attn_hidden_dim=cross_attn_hidden_dim
        )
        self.depression_predictor = DepressionPredictor(
            input_dim=self.feature_extractor.output_dim,
            hidden_dim=fc_hidden_dim
        )
        self.grl = GradientReversal(alpha=lambda_grl)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=self.feature_extractor.output_dim,
            hidden_dim=fc_hidden_dim,
            n_domains=n_domains
        )

    def forward(self, x_audio, x_text):
        features = self.feature_extractor(x_audio, x_text)
        depression_logits = self.depression_predictor(features)

        if self.training and self.generalization:
            domain_logits = self.domain_discriminator(self.grl(features))
            return depression_logits, domain_logits
        else:
            return depression_logits
