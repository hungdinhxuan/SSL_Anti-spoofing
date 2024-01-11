import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn as nn
from model import Model
import os
import librosa
from torch import nn
from lightning.pytorch import LightningModule
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor.quantization import fit
from neural_compressor.data import DataLoader, Datasets
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.utils.pytorch import load

# Quantization neural compressor
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_model_size(model):
    torch.save(model.state_dict(), 'temp_saved_model.pt')
    model_size_in_mb = os.path.getsize('temp_saved_model.pt') >> 20
    os.remove('temp_saved_model.pt')
    return model_size_in_mb
device = "cpu"
dummy_input = torch.rand(1, 60000)

model = Model(None,device)
model =nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('/datab/hungdx/SSL_Anti-spoofing/Best_LA_model_for_DF.pth',map_location=device))
model.eval()


class AASIST_SSL_Wrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.relu = torch.nn.ReLU()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    
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
                
        return self.dequant(output_probs[0][0])

audio,_ = librosa.load('/datab/hungdx/SSL_Anti-spoofing/commonvoice/test/clips/common_voice_en_19698109.wav', sr=16000)
_model = AASIST_SSL_Wrapper(model)
_model.eval()
with torch.no_grad():
    print(_model(Tensor(audio)))

# dataset = Datasets("pytorch")["dummy"](shape=(1, 60000))
# dataloader = DataLoader(framework="pytorch", dataset=dataset)

# q_model = fit(
#     model=_model,
#     conf=PostTrainingQuantConfig(),
#     calib_dataloader=dataloader,
# )
# q_model.save("./saved_model/")



int8_model = load('./saved_model/', _model)

print(int8_model(Tensor(audio)))

# print("original model size is {:.2f} MB".format(get_model_size(_model)))
# quantized_model = torch.quantization.quantize_dynamic(_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
# script_module = torch.jit.script(_model(dummy_input))
# dummy_input = torch.rand(1, 60000)
# traced_module = torch.jit.trace(_model, dummy_input)
# optimized_model = optimize_for_mobile(traced_module)
# optimized_model._save_for_lite_interpreter("aasist_ssl_w2v_fairseq.ptl")
# print("quantized model size is {:.2f} MB".format(get_model_size(optimized_model)))
    
