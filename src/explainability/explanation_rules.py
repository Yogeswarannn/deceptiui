def generate_explanation(class_name, confidence, text_evidence_list):
    """
    Generates a human-readable explanation combining class, confidence, and evidence.
    """
    # Base templates based on class
    templates = {
        "hard_to_close": "The interface makes dismissing or exiting an experience unnecessarily difficult.",
        "default_choice": "The interface appears to steer the user toward a pre-selected or nudged option.",
        "false_hierarchy": "One option appears to receive stronger visual emphasis than the alternatives.",
        "nagging": "The interface features repeated prompts or interruptions pushing the user toward an action.",
        "privacy_zuckering": "The interface may encourage the user to share more personal information than necessary."
    }
    
    # Normalize class name for dictionary lookup (e.g., "Hard to Close" -> "hard_to_close")
    class_key = class_name.lower().replace(" ", "_")
    base_explanation = templates.get(class_key, f"The interface shows signs of {class_name}.")
    
    explanation_parts = [base_explanation]
    
    # Add textual evidence explanation
    if text_evidence_list:
        phrases = [item['phrase'] for item in text_evidence_list[:3]]
        
        if len(phrases) == 1:
            phrase_str = f'"{phrases[0]}"'
        elif len(phrases) == 2:
            phrase_str = f'"{phrases[0]}" and "{phrases[1]}"'
        else:
            phrase_str = f'"{phrases[0]}", "{phrases[1]}", and "{phrases[2]}"'
            
        explanation_parts.append(f"Specific text such as {phrase_str} contributed significantly to this prediction.")
    
    # Add confidence phrasing
    if confidence >= 0.9:
        explanation_parts.append("The model is highly confident in this detection based on the combined visual and textual features.")
    elif confidence >= 0.7:
        explanation_parts.append("The model has moderate confidence in this detection.")
    else:
        explanation_parts.append("This is a low-confidence detection; the interface may only contain subtle deceptive patterns.")
        
    return " ".join(explanation_parts)
