import os

# Set CUDA_VISIBLE_DEVICES to an empty string to hide all GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load audio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import torch
import torchaudio
import librosa.display
from model import Model
from torch import nn
import time
from data_utils_SSL import pad
from torch import Tensor
import librosa
import psutil

#torch.set_num_threads(1)

def load_audio(path):
    audio, sample_rate = torchaudio.load(path)
    return audio, sample_rate

device = 'cpu'
model_path = './Best_LA_model_for_DF.pth'
print('Device: {}'.format(device))

model = Model(None, device=device).to(device)
nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
print('nb_params:',nb_params)
model.load_state_dict(torch.load(model_path,map_location=device), strict=False)
print('Model loaded : {}'.format(model_path))

model.eval()

def detect_deepfake(audio_data, model, segment_duration=4.0, sampling_rate=16000):
    segment_length = int(segment_duration * sampling_rate)
    total_segments = len(audio_data) // segment_length
    predictions = []
    latencies = []
    with torch.no_grad():
        for i in range(total_segments):
            segment = audio_data[i * segment_length:(i + 1) * segment_length]            
            segment = pad(segment, 66800)
            segment = Tensor(segment).unsqueeze(0).to(device)
            print(segment.shape)
            start_time = time.time()
            out = model(segment)
            
            end_time = time.time()
            output_probs = torch.sigmoid(out)
            predictions.append(output_probs)
            # Calculate latency
            latency = abs(end_time - start_time) * 1000
            latencies.append(latency)
    print(f'avg_latency: {np.mean(latencies)} ms on Snap Highcost')
    return predictions, latencies

# Run the deepfake detection simulation
fake_audio_jp = librosa.load('/datab/hungdx/conformer-based-classifier-for-anti-spoofing/000126_SeamlessM4T-TTS_jpn.wav', sr=16000)

# Repeat audio 50 times
fake_audio2 = np.tile(fake_audio_jp[0], 1)

predictions, latencies = detect_deepfake(fake_audio2, model)