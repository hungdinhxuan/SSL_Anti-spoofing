import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn as nn
from aasist import Model
from transformers import Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModel
import os
import librosa


os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_model_size(model):
    torch.save(model.state_dict(), 'temp_saved_model.pt')
    model_size_in_mb = os.path.getsize('temp_saved_model.pt') >> 20
    os.remove('temp_saved_model.pt')
    return model_size_in_mb
device = "cpu"
dummy_input = torch.rand(1, 60000)


wav2vec2_xls_r_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m")
model = import_huggingface_model(wav2vec2_xls_r_model)
num_params = sum([param.view(-1).size()[0] for param in model.parameters()])
print(f"Model {num_params} parameters")


# # Convert the model to torchaudio format, which supports TorchScript.
# wav2vec2_xls_r_model = import_huggingface_model(wav2vec2_xls_r_model)



# # Remove weight normalization which is not supported by quantization.

# wav2vec2_xls_r_model.eval()
# Load AASIST-SSL model

# model = Model(wav2vec2_xls_r_model)
# model =nn.DataParallel(model).to(device)
# model.load_state_dict(torch.load('/datab/hungdx/SSL_Anti-spoofing/models/model_DF_weighted_CCE_100_32_1e-06/epoch_0.pth',map_location=device))
# model.eval()

# class AASIST_SSL_Wrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def pad(self, x, max_len: int = 64600) -> Tensor:
#         x_len = x.shape[0]
#         if x_len >= max_len:
#             return x[:max_len]
#         # need to pad
#         num_repeats = int(max_len / x_len)+1
#         padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
#         return padded_x
			
    
#     def forward(self, wavforms: Tensor) -> Tensor:
#         wav_padded = self.pad(wavforms).unsqueeze(0)
#         out = self.model(wav_padded)
#         output_probs = torch.sigmoid(out)
#         return output_probs[0][0]

# audio,_ = librosa.load('/datab/hungdx/SSL_Anti-spoofing/commonvoice/test/clips/common_voice_en_19698109.wav', sr=16000)
# _model = AASIST_SSL_Wrapper(model)
# _model.eval()
# with torch.no_grad():
#     print(_model(Tensor(audio)))
# print("original model size is {:.2f} MB".format(get_model_size(_model)))

# quantized_model = torch.quantization.quantize_dynamic(_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
# dummy_input = torch.rand(1, 60000)
# traced_module = torch.jit.trace(quantized_model, dummy_input)
# optimized_model = optimize_for_mobile(traced_module)
# optimized_model._save_for_lite_interpreter("aasist_ssl_w2v_hugging_face.ptl")
# print("quantized model size is {:.2f} MB".format(get_model_size(optimized_model)))