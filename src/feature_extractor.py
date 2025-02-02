from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch


class AudioFeatureExtractor:
    def __init__(self, model_name, device):
        models = {
            'Wav2Vec2': 'facebook/wav2vec2-base-960h',
            'HuBERT': 'facebook/hubert-large-ls960-ft'
        }
        self.model_name = models[model_name]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.feature_dim = self.model.config.hidden_size
        self.sample_rate = 16000

    def __call__(self, audio_segments):
        segments_features = []
        for segment in audio_segments:
            inputs = self.processor(segment, sampling_rate=self.sample_rate, return_tensors='pt', padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self.model(**inputs).last_hidden_state
            segments_features.append(features)
        features = torch.cat(segments_features, dim=1)
        return features


class TextFeatureExtractor:
    def __init__(self, model_name, device):
        models = {
            'BERT': 'bert-base-uncased',
            'ItalianBERT': 'dbmdz/bert-base-italian-cased'
        }
        self.model_name = models[model_name]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.feature_dim = self.model.config.hidden_size

    def __call__(self, text_segments):
        segments_features = []
        for segment in text_segments:
            inputs = self.tokenizer(segment, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self.model(**inputs).last_hidden_state
            segments_features.append(features)
        features = torch.cat(segments_features, dim=1)
        return features
