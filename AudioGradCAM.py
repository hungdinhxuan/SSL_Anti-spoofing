import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import matplotlib.cm as cm


class AudioGradCAM:
    def __init__(self, model, target_layer, device):
        # Store the model, target layer, and device references
        self.model = model.to(device)
        self.target_layer = target_layer
        self.device = device

        # Placeholder for feature maps and gradients
        self.feature_maps = None
        self.gradients = None

        # Register hooks to the target layer to capture feature maps and gradients

        # This means that whenever a forward pass is made through self.target_layer,
        # after computing its output, the function save_feature_maps will be called.
        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_feature_maps)

        # After the gradients for self.target_layer are computed during backpropagation,
        # the save_gradients function will be invoked.
        self.backward_hook = self.target_layer.register_backward_hook(
            self.save_gradients)

    def save_feature_maps(self, module, input, output):
        """Save feature maps during forward pass"""
        self.feature_maps = output.detach()
        print(f'feature maps(Activations) size: {self.feature_maps.size()}')

    def save_gradients(self, module, grad_in, grad_out):
        """Save gradients during backward pass"""
        self.gradients = grad_out[0].detach()
        print(f'Gradients size: {self.gradients[0].size()}')

    def compute_weights(self):
        """Compute the weights by global average pooling the gradients"""
        return torch.mean(self.gradients, dim=[2], keepdim=True)

    def __call__(self, input_tensor, score_forward=None, target_class=None):
        # Set model to evaluation mode
        self.model.eval()

        # Ensure input_tensor is on the right device
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        scores = score_forward

        # If no specific class is specified, use the class with the highest score
        if target_class is None:
            target_class = scores.view(-1).argmax().item()
            print(target_class)

        # Zero gradients everywhere
        self.model.zero_grad()

        # Backward pass with only the specific class's score
        #  Assuming scores has a shape like (batch_size, 1, num_classes),
        # this line is selecting the score for target_class for the first sample in the batch.
        scores[0, 0, target_class].backward(retain_graph=True)

        # Calculate weights alpha by global average pooling the gradients
        weights = self.compute_weights()

        print(
            f'weights shape after Calculate weights alpha by global average pooling the gradients: {weights.shape}')
        print(f'feature_maps shape before grad-cam: {self.feature_maps.shape}')

        # Compute the Grad-CAM map
        grad_cam_map = torch.sum((weights * self.feature_maps), dim=1)

        print(f'Grad-CAM map shape before Relu: {grad_cam_map.shape}')

        # ReLU to only keep positive values
        grad_cam_map = torch.relu(grad_cam_map)

        return grad_cam_map

    def visualize_with_spectrogram_and_heatmap(waveform, heatmap, sr=16000):
        # Ensure input_audio and heatmap are on CPU
        # Squeeze the tensors to ensure they're at most 2D
        waveform = waveform.detach().cpu().numpy()
        heatmap = heatmap.cpu().numpy()

        # Compute the mel spectrogram
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Interpolate the heatmap to match spectrogram size
        heatmap_resized = np.interp(np.linspace(
            0, 1, log_S.shape[1]), np.linspace(0, 1, heatmap.shape[0]), heatmap)

        plt.figure(figsize=(15, 10))

        # Plot raw waveform
        plt.subplot(3, 1, 1)
        plt.plot(waveform, color='b')
        plt.title("Raw Waveform")

        # Plot mel spectrogram
        plt.subplot(3, 1, 2)
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+02.0f dB')
        plt.title('Mel spectrogram')

        # Overlay Grad-CAM heatmap on the spectrogram
        plt.subplot(3, 1, 3)
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+02.0f dB')
        plt.imshow(np.repeat(
            heatmap_resized[:, np.newaxis], log_S.shape[0], axis=1), aspect='auto', cmap='inferno', alpha=0.5)
        plt.title('Mel spectrogram with Grad-CAM')

        plt.tight_layout()
        plt.show()

    def visualize1D(self, waveform, heatmap):
        # Convert tensors to numpy

        heatmap = heatmap.cpu().numpy()
        waveform = waveform.cpu().numpy()
        print(f'heatmap: {heatmap}')
        # Interpolate heatmap to match waveform size

        print(f'Interpolate f param 1: {np.linspace(0, 1, waveform.shape[0])}')
        print(f'Interpolate f param 2: {np.linspace(0, 1, heatmap.shape[0])}')
        heatmap_resized = np.interp(np.linspace(
            0, 1, waveform.shape[0]), np.linspace(0, 1, heatmap.shape[0]), heatmap)

        # Normalize the heatmap for coloring
        norm = plt.Normalize(heatmap_resized.min(), heatmap_resized.max())
        print(
            f'heatmap_resized.min() {heatmap_resized.min()}, heatmap_resized.max() {heatmap_resized.max()}')
        plt.figure(figsize=(15, 5))

        # Plot raw waveform
        plt.plot(waveform, label='Waveform', color='blue', alpha=0.5)

        # Overlay Grad-CAM heatmap using scatter for colored representation
        plt.scatter(np.arange(len(waveform)), waveform,
                    c=heatmap_resized, cmap=cm.jet, label='Importance', norm=norm)

        plt.title('Waveform with Grad-CAM Overlay')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Importance Strength')

        plt.legend()

        plt.tight_layout()
        plt.show()

    def clear_hooks(self):
        """Remove hooks from the model"""
        self.forward_hook.remove()
        self.backward_hook.remove()
