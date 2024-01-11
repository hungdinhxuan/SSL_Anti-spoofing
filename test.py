from model import Model
import torch.nn as nn
import numpy as np
import torch
from torch import Tensor
import librosa
import os

# Set CUDA_VISIBLE_DEVICES to an empty string to hide all GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ''
class AASIST_SSL_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def pad(self, x, max_len: int = 64600) -> Tensor:
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
        return padded_x
			
    
    def forward(self, wavforms: Tensor) -> Tensor:
        wav_padded = self.pad(wavforms).unsqueeze(0)
        out = self.model(wav_padded)
        output_probs = torch.sigmoid(out)
        return output_probs[0][0]
    
device = "cpu"
model = Model(None, device)
model =nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('./Best_LA_model_for_DF.pth',map_location=device))
model.eval()
print(model.module.ssl_model.state_dict().keys())

# _model = AASIST_SSL_Wrapper(model)
# _model.eval()

# with torch.no_grad():

#     audio,_ = librosa.load('/datab/hungdx/SSL_Anti-spoofing/common_voice_en_1075.wav', sr=16000)
#     print(_model(Tensor(audio)))
