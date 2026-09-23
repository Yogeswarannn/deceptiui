import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import base64
from io import BytesIO

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_image, text_features, target_class=None):
        # Forward pass
        self.model.zero_grad()
        model_output = self.model(input_image, text_features)
        
        # Target for backprop
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1).item()
            
        target = model_output[0][target_class]
        
        # Backward pass
        target.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap
        
def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """
    Overlays the heatmap on the original image.
    original_image: PIL Image
    heatmap: 2D numpy array (0 to 1)
    """
    width, height = original_image.size
    
    # Resize heatmap to match original image using PIL
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize((width, height), resample=Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_img)
    
    # Apply colormap
    colormap = cm.get_cmap('jet') if hasattr(cm, 'get_cmap') else __import__('matplotlib').colormaps['jet']
    heatmap_colored = colormap(heatmap_resized)[:, :, :3] * 255.0
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    heatmap_pil = Image.fromarray(heatmap_colored).convert("RGBA")
    
    # Create an alpha mask based on the heatmap intensity
    alpha_mask = (heatmap_resized * 255 * alpha).astype(np.uint8)
    heatmap_pil.putalpha(Image.fromarray(alpha_mask))
    
    # Paste onto original image
    original_rgba = original_image.convert("RGBA")
    overlayed_img = Image.alpha_composite(original_rgba, heatmap_pil)
    
    return overlayed_img.convert("RGB")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
