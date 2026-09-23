import torch
import numpy as np

def extract_text_evidence(model, input_image, text_features, vectorizer, target_class=None, top_k=5):
    """
    Extracts text evidence using Gradient * Input method.
    """
    model.eval()
    
    # Ensure text_features requires grad
    if not text_features.requires_grad:
        text_features.requires_grad_(True)
        
    model.zero_grad()
    
    # Forward pass
    output = model(input_image, text_features)
    
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()
        
    # Get the target logit
    target_logit = output[0][target_class]
    
    # Backward pass
    target_logit.backward()
    
    # Get gradients for text features
    gradients = text_features.grad.cpu().numpy()[0]
    input_features = text_features.detach().cpu().numpy()[0]
    
    # Gradient * Input attribution
    attributions = gradients * input_features
    
    # Get top k indices with highest positive attribution
    top_indices = np.argsort(attributions)[::-1]
    
    # Filter to only indices that were actually present in the text
    # (i.e. input feature > 0) and have positive attribution
    feature_names = vectorizer.get_feature_names_out()
    
    evidence_phrases = []
    for idx in top_indices:
        if len(evidence_phrases) >= top_k:
            break
            
        if input_features[idx] > 0 and attributions[idx] > 0:
            phrase = feature_names[idx]
            score = float(attributions[idx])
            evidence_phrases.append({
                "phrase": phrase,
                "importance": score
            })
            
    return evidence_phrases
