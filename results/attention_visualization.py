import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer
)
from bertviz import head_view, model_view
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset, clean_text, set_seed
from results.evaluate_fine_tuned import load_zulu_dataset, load_pcm_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"attention_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_models_and_tokenizers(models_dir, emotion_labels):
    logger.info("Loading fine-tuned mBERT model...")
    mb_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    mb_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(models_dir, "mbert_english"),
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification",
        output_attentions=True
    )
    
    logger.info("Loading fine-tuned AfriBERTa model...")
    af_tokenizer = XLMRobertaTokenizer.from_pretrained("castorini/afriberta_base")
    af_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(models_dir, "afriberta_english"),
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification",
        output_attentions=True
    )
    
    return {
        'mb_model': mb_model,
        'mb_tokenizer': mb_tokenizer,
        'af_model': af_model,
        'af_tokenizer': af_tokenizer
    }

def get_attention_for_text(model, tokenizer, text, emotion_labels, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()[0]
    
    attentions = outputs.attentions
    
    result = {
        'text': text,
        'predictions': {label: bool(pred) for label, pred in zip(emotion_labels, predictions)},
        'logits': logits.cpu().numpy()[0].tolist(),
        'token_ids': inputs['input_ids'].cpu().numpy()[0].tolist(),
        'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
        'attentions': [att.cpu().numpy() for att in attentions]
    }
    
    return result

def visualize_attention(model_name, tokenizer, model, texts, emotion_labels, device, output_dir):
    """Generate and save attention visualizations for a set of example texts"""
    logger.info(f"Generating attention visualizations for {model_name}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, text in enumerate(texts):
        # Get attention and predictions
        result = get_attention_for_text(model, tokenizer, text, emotion_labels, device)
        
        # Log predictions
        logger.info(f"Example {i+1}: {text}")
        logger.info(f"Predicted emotions: {[e for e, v in result['predictions'].items() if v]}")
        
        # Create HTML visualization files using bertviz
        tokens = result['tokens']
        attention = torch.tensor(result['attentions'])  # Convert back to tensor for bertviz
        
        # Generate single-head view (for the first layer, first head as example)
        head_view_html = head_view(
            attention[0:1],  # First layer
            tokens,
            jupyter_backend=False  # Using matplotlib backend for saving
        )
        
        # Save HTML files
        output_file = os.path.join(output_dir, f"{model_name}_{i+1}_head_view.html")
        with open(output_file, "w") as f:
            f.write(head_view_html)
        logger.info(f"Saved visualization to {output_file}")
        
        # Optional: Save model view visualization (shows all layers/heads)
        model_view_html = model_view(
            attention,
            tokens,
            jupyter_backend=False
        )
        output_file = os.path.join(output_dir, f"{model_name}_{i+1}_model_view.html")
        with open(output_file, "w") as f:
            f.write(model_view_html)
        logger.info(f"Saved full model visualization to {output_file}")

def main():
    set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    zulu_dataset, emotion_labels = load_zulu_dataset()
    pcm_dataset, _ = load_pcm_dataset()
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved-models")
    
    models = load_models_and_tokenizers(models_dir, emotion_labels)
    
    viz_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)
    
    zulu_examples = [
        zulu_dataset['test'][i]['text'] for i in range(5)
    ]
    
    pcm_examples = [
        pcm_dataset['test'][i]['text'] for i in range(5)
    ]
    
    # Visualize attention for mBERT on Zulu
    visualize_attention(
        "mBERT_Zulu",
        models['mb_tokenizer'],
        models['mb_model'],
        zulu_examples,
        emotion_labels,
        device,
        os.path.join(viz_output_dir, "mbert_zulu")
    )
    
    # Visualize attention for AfriBERTa on Zulu
    visualize_attention(
        "AfriBERTa_Zulu",
        models['af_tokenizer'],
        models['af_model'],
        zulu_examples,
        emotion_labels,
        device,
        os.path.join(viz_output_dir, "afriberta_zulu")
    )
    
    # Visualize attention for mBERT on Nigerian Pidgin English
    visualize_attention(
        "mBERT_PCM",
        models['mb_tokenizer'],
        models['mb_model'],
        pcm_examples,
        emotion_labels,
        device,
        os.path.join(viz_output_dir, "mbert_pcm")
    )
    
    # Visualize attention for AfriBERTa on Nigerian Pidgin English
    visualize_attention(
        "AfriBERTa_PCM",
        models['af_tokenizer'],
        models['af_model'],
        pcm_examples,
        emotion_labels,
        device,
        os.path.join(viz_output_dir, "afriberta_pcm")
    )
    
    logger.info("Attention visualization completed.")

if __name__ == "__main__":
    main()
