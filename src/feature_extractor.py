from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch


class AudioFeatureExtractor:
    def __init__(self, model_name):
        models = {
            'Wav2Vec2': 'facebook/wav2vec2-base-960h',
            'HuBERT': 'facebook/hubert-large-ls960-ft'
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = models[model_name]
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.feature_dim = self.model.config.hidden_size
        self.sample_rate = 16000

    def __call__(self, audio_segments):
        segments_features = []
        for segment in audio_segments:
            segment = torch.tensor(segment, dtype=torch.float32)
            inputs = self.processor(segment, sampling_rate=self.sample_rate, return_tensors='pt', padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self.model(**inputs).last_hidden_state
            segments_features.append(features)
        features = torch.cat(segments_features, dim=1)
        return features


class TextFeatureExtractor:
    def __init__(self, model_name):
        models = {
            'BERT': 'bert-base-uncased',
            'ItalianBERT': 'dbmdz/bert-base-italian-cased'
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = models[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
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


class LabelTransformer:
    def __call__(self, y):
        return torch.tensor(y, dtype=torch.float32).unsqueeze(0)
