from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from torch.nn.utils.rnn import pad_sequence
import torch
from librosa import power_to_db
from librosa.feature import melspectrogram, mfcc, delta
import numpy as np
import gc


class AudioFeatureExtractor:
    def __init__(self, model_name, segment_duration, device, sample_rate=16000):
        self.model_name = model_name
        self.segment_duration = segment_duration
        self.device = device
        self.sample_rate = sample_rate
        self.init_extractor()

    def init_extractor(self):
        if self.model_name == 'MelSpec':
            self.n_mels = 128
            self.feature_dim = self.n_mels * 3
            self.extractor = self.traditional_extractor
        elif self.model_name == 'MFCC':
            self.n_mfcc = 13
            self.feature_dim = self.n_mfcc * 3
            self.extractor = self.traditional_extractor
        else:
            models = {
                'Wav2Vec2': 'facebook/wav2vec2-base-960h',
                'HuBERT': 'facebook/hubert-large-ls960-ft',
            }
            self.model_name = models[self.model_name]
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.concat_dim = 0 if self.segment_duration else 1
            self.feature_dim = self.model.config.hidden_size
            self.extractor = self.transformer_extractor

    def transformer_extractor(self, audio_segments):
        segments_features = []
        for segment in audio_segments:
            segment = torch.tensor(segment, dtype=torch.float32)
            input_features = self.processor(segment, sampling_rate=self.sample_rate, return_tensors='pt', padding=True)
            input_features = {key: value.to(self.device) for key, value in input_features.items()}
            with torch.no_grad():
                features = self.model(**input_features).last_hidden_state.cpu()
            segments_features.append(features)

            del input_features, features
            torch.cuda.empty_cache()
            gc.collect()
        return torch.cat(segments_features, dim=self.concat_dim)

    def traditional_extractor(self, audio_segments, n_fft=4096, hop_length=512):
        segments_features = []
        for segment in audio_segments:
            if self.model_name == 'MelSpec':
                # extract log-mel spectrogram features
                features = power_to_db(
                    melspectrogram(
                        y=segment,
                        sr=self.sample_rate,
                        n_mels=self.n_mels,
                        hop_length=hop_length,
                        n_fft=n_fft
                    ), ref=np.max
                )
            elif self.model_name == 'MFCC':
                # extract MFCC features
                features = mfcc(
                    y=segment,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    hop_length=hop_length,
                    n_fft=n_fft
                )
            delta_features = torch.tensor(delta(features), dtype=torch.float32)
            delta2_features = torch.tensor(delta(features, order=2), dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.float32)

            features = torch.cat([features.T, delta_features.T, delta2_features.T], dim=-1)
            segments_features.append(features)

        return torch.stack(segments_features)

    def __call__(self, audio_segments):
        return self.extractor(audio_segments)


class TextFeatureExtractor:
    def __init__(self, model_name, segment_duration, device):
        models = {
            'BERT': 'bert-base-uncased',
            'ItalianBERT': 'dbmdz/bert-base-italian-xxl-cased',
            'XLMRoBERTa': 'FacebookAI/xlm-roberta-large'
        }
        self.segment_duration = segment_duration
        self.device = device
        self.model_name = models[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.feature_dim = self.model.config.hidden_size

    def __call__(self, text_segments):
        segments_features = []
        for segment in text_segments:
            input_features = self.tokenizer(segment, return_tensors='pt', padding=True, truncation=True)
            input_features = {key: value.to(self.device) for key, value in input_features.items()}
            with torch.no_grad():
                features = self.model(**input_features).last_hidden_state.cpu()
            segments_features.append(features)

            del input_features, features
            torch.cuda.empty_cache()
            gc.collect()

        if self.segment_duration:
            return pad_sequence([seg.squeeze(0) for seg in segments_features], batch_first=True, padding_value=0.0)
        else:
            return torch.cat(segments_features, dim=1)


class LabelTransformer:
    def __call__(self, y):
        return torch.tensor(y, dtype=torch.float32).unsqueeze(0)


class Transcriber:
    def __init__(self, device, model_name='openai/whisper-large-v3'):
        torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)

        self.pipe = pipeline(
            'automatic-speech-recognition',
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={'return_timestamps': True}
        )

    def __call__(self, x):
        if isinstance(x, list):
            return [transcript['text'] for transcript in self.pipe(x, batch_size=8)]
        elif isinstance(x, np.ndarray):
            return self.pipe(x)['text']
