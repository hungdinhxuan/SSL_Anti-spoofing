import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers import Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from aasist import SSLModelCustom
from torch import Tensor

input_audio, sample_rate = librosa.load("/datab/hungdx/SSL_Anti-spoofing/common_voice_en_1075.wav",  sr=16000)

model_name = "facebook/wav2vec2-xls-r-300m"

wav2vec2_xls_r_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m")

# Convert the model to torchaudio format, which supports TorchScript.
wav2vec2_xls_r_model = import_huggingface_model(wav2vec2_xls_r_model)


ssl_model = SSLModelCustom(wav2vec2_xls_r_model)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
nb_params2 = sum([param.view(-1).size()[0] for param in ssl_model.parameters()])
print(f"Model {model_name} Wav2Vec2FeatureExtractor has {nb_params} parameters")
print(f"Model {model_name} ssl_model has {nb_params} parameters")
i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
with torch.no_grad():
  o= model(i.input_values)
  
print(o.keys())
print(o.last_hidden_state.shape)
print(o.extract_features.shape)

print(ssl_model(Tensor(input_audio).unsqueeze(0)).shape)