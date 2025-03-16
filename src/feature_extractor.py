from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
import numpy as np
import gc


class AudioFeatureExtractor:
    def __init__(self, model_name, segment_duration, device):
        models = {
            'Wav2Vec2': 'facebook/wav2vec2-base-960h',
            'HuBERT': 'facebook/hubert-large-ls960-ft'
        }
        self.segment_duration = segment_duration
        self.device = device
        self.model_name = models[model_name]
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.feature_dim = self.model.config.hidden_size
        self.sample_rate = 16000

    def __call__(self, audio_segments):
        segments_features = []
        for segment in audio_segments:
            segment = torch.tensor(segment, dtype=torch.float32)
            input_features = self.processor(segment, sampling_rate=self.sample_rate, return_tensors='pt', padding=True)
            input_features = {key: value.to(self.device) for key, value in input_features.items()}
            with torch.no_grad():
                features = self.model(**input_features).last_hidden_state.cpu()
            segments_features.append(features)

            # Free up the memory from GPU
            del input_features, features, segment
            torch.cuda.empty_cache()
            gc.collect()

        if self.segment_duration:
            return segments_features
        else:
            return torch.cat(segments_features, dim=1)


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

            # Free up the memory from GPU
            del input_features, features, segment
            torch.cuda.empty_cache()
            gc.collect()

        if self.segment_duration:
            return segments_features
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
            device=device
        )

    def __call__(self, x):
        if isinstance(x, list):
            return [transcript['text'] for transcript in self.pipe(x, batch_size=8)]
        elif isinstance(x, np.ndarray):
            return self.pipe(x)['text']
