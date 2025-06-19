"""
Use SHAP (SHapley Additive exPlanations) to explain predictions of fine-tuned
mBERT and AfriBERTa models in zero-shot settings (trained on English, tested on Zulu and Pidgin).
This script specifically handles zero-shot SHAP analysis with improved error handling and debugging.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer
)
import logging
from datetime import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_and_preprocess_data, set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"zero_shot_shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_models_and_tokenizers(models_dir, emotion_labels):
    """Load the fine-tuned models and tokenizers"""
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info("Loading fine-tuned mBERT model...")
    mb_model_path = os.path.join(models_dir, "mbert_english")
    if not os.path.exists(mb_model_path):
        logger.warning(f"Model path {mb_model_path} doesn't exist. Will try to load from Hugging Face.")
    
    try:
        mb_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        mb_model = AutoModelForSequenceClassification.from_pretrained(
            mb_model_path if os.path.exists(mb_model_path) else "bert-base-multilingual-cased",
            num_labels=len(emotion_labels),
            problem_type="multi_label_classification"
        )    
    except Exception as e:
        logger.error(f"Error loading mBERT model: {e}")
        traceback.print_exc()
        mb_tokenizer = None
        mb_model = None
    
    # AfriBERTa
    logger.info("Loading fine-tuned AfriBERTa model...")
    af_model_path = os.path.join(models_dir, "afriberta_english")
    if not os.path.exists(af_model_path):
        logger.warning(f"Model path {af_model_path} doesn't exist. Will try to load from Hugging Face.")
    
    try:
        af_tokenizer = XLMRobertaTokenizer.from_pretrained("castorini/afriberta_base")
        af_model = AutoModelForSequenceClassification.from_pretrained(
            af_model_path if os.path.exists(af_model_path) else "castorini/afriberta_base",
            num_labels=len(emotion_labels),
            problem_type="multi_label_classification"
        )
    except Exception as e:
        logger.error(f"Error loading AfriBERTa model: {e}")
        traceback.print_exc()
        af_tokenizer = None
        af_model = None
    
    return {
        'mb_model': mb_model,
        'mb_tokenizer': mb_tokenizer,
        'af_model': af_model,
        'af_tokenizer': af_tokenizer
    }

class TransformerExplainer:
    def __init__(self, model, tokenizer, device, model_name=""):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        
        if model is not None:
            self.model.to(device)
            self.model.eval()
    
    def predict(self, texts):
        """Predict for a batch of texts and return logits"""
        if self.model is None or self.tokenizer is None:
            logger.error(f"Cannot predict: model or tokenizer is None for {self.model_name}")
            return np.zeros((len(texts), 6))  
        
    
        if not isinstance(texts, list):
            if isinstance(texts, str):
                texts = [texts]  
            else:
                texts = list(map(str, texts))  
        else:
            
            texts = [str(text) for text in texts]
        
        try:
          
            inputs = self.tokenizer(texts, 
                                return_tensors="pt", 
                                padding=True,
                                truncation=True, 
                                max_length=128)
            
       
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            
            return probs
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            traceback.print_exc()
            return np.zeros((len(texts), 6))  
    
    def predict_for_shap(self, texts):
        """A wrapper for predict that returns a numpy array"""
        result = self.predict(texts)
        
        logger.debug(f"Predict_for_shap called with {len(texts)} texts")
        logger.debug(f"Prediction shape: {result.shape}")
        return result

def load_zulu_dataset(num_examples=5):
    """Load Zulu dataset with the given number of examples"""
    logger.info("Loading Zulu dataset...")
    data = load_and_preprocess_data()
    zulu_dataset = data['zulu_dataset']
    emotion_labels = data['emotion_labels']

    texts = [str(text).strip() for text in zulu_dataset['test']['text'][:num_examples]]
    
    logger.info(f"Loaded {len(texts)} Zulu examples for SHAP analysis")
    
    return texts, emotion_labels

def load_pidgin_dataset(num_examples=5):
    """Load Nigerian Pidgin dataset with the given number of examples"""
    logger.info("Loading Nigerian Pidgin dataset...")
    data = load_and_preprocess_data()
    pidgin_dataset = data['pidgin_dataset']
    emotion_labels = data['emotion_labels']
    
    texts = [str(text).strip() for text in pidgin_dataset['test']['text'][:num_examples]]
    
    logger.info(f"Loaded {len(texts)} Nigerian Pidgin examples for SHAP analysis")
    
    return texts, emotion_labels

def generate_basic_shap_plots(model_name, language, texts, predictions, emotion_labels, output_dir):
    """Generate basic plots showing model predictions and heatmap-style visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    

    for i, text in enumerate(texts):
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(emotion_labels))
      
        preds = predictions[i]
        
      
        plt.barh(y_pos, preds, align='center', alpha=0.5)
        plt.yticks(y_pos, emotion_labels)
        plt.xlabel('Prediction Score')
        plt.title(f'{model_name} on {language} - Example {i+1}')
        
        plt.figtext(0.5, 0.95, f"Text: {text[:100]}{'...' if len(text) > 100 else ''}", 
                   wrap=True, horizontalalignment='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"example_{i+1}_predictions.png"))
        plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(predictions, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Prediction Score')
    plt.xticks(np.arange(len(emotion_labels)), emotion_labels, rotation=45)
    plt.yticks(np.arange(len(texts)), [f"Ex {i+1}" for i in range(len(texts))])
    plt.title(f'{model_name} on {language} - Prediction Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_heatmap.png"))
    plt.close()
    
    with open(os.path.join(output_dir, "predictions.txt"), 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts):
            f.write(f"Example {i+1}: {text}\n")
            for j, emotion in enumerate(emotion_labels):
                score = predictions[i][j]
                f.write(f"  {emotion}: {score:.4f}\n")
            f.write("\n")

def safe_shap_explainer(explainer, texts, max_retries=3, batch_size=2):
    """Safe SHAP explainer that tries different strategies to get SHAP values"""
    
    try:
        logger.info(f"Attempting standard SHAP calculation with {len(texts)} examples")
        masker = shap.maskers.Text(explainer.tokenizer)
        shap_explainer = shap.Explainer(explainer.predict_for_shap, masker=masker)
        with np.errstate(invalid='ignore'):
            shap_values = shap_explainer(texts)
        return shap_values
    except Exception as e:
        logger.warning(f"Standard SHAP calculation failed: {e}")
    
    for batch_size in [2, 1]:
        try:
            logger.info(f"Trying batch-by-batch SHAP calculation with batch size {batch_size}")
            masker = shap.maskers.Text(explainer.tokenizer)
            shap_explainer = shap.Explainer(explainer.predict_for_shap, masker=masker)
            
            all_values = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_texts)} texts")
                
                with np.errstate(invalid='ignore'):
                    batch_values = shap_explainer(batch_texts)
                all_values.append(batch_values)
      
            if all_values:
                shap_values = all_values[0]
                return shap_values
        except Exception as e:
            logger.warning(f"Batch SHAP calculation with size {batch_size} failed: {e}")
    
    if len(texts) > 0:
        try:
            shortest_text = min(texts, key=len)
            logger.info(f"Trying with a single, short example: '{shortest_text[:20]}...'")
            masker = shap.maskers.Text(explainer.tokenizer)
            shap_explainer = shap.Explainer(explainer.predict_for_shap, masker=masker)
            with np.errstate(invalid='ignore'):
                shap_values = shap_explainer([shortest_text])
            return shap_values
        except Exception as e:
            logger.warning(f"Single example SHAP calculation failed: {e}")
    
    logger.error("All SHAP calculation methods failed")
    return None

def generate_token_importance_summary(shap_values, texts, emotion_labels, model_name, language, output_dir):
    """Generate token importance summary from SHAP values"""
    if shap_values is None:
        logger.error("Cannot generate token importance summary: SHAP values are None")
        return
    
    try:
  
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "token_importance.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Token Importance Analysis for {model_name} on {language}\n\n")
            
            for i, text in enumerate(texts):
                f.write(f"Example {i+1}: {text}\n\n")
                
                if hasattr(shap_values, 'values') and hasattr(shap_values, 'data'):
                    try:
                      
                        for e_idx, emotion in enumerate(emotion_labels):
                            f.write(f"  {emotion} emotion:\n")
                           
                            if len(shap_values.values.shape) >= 3:
                            
                                if i < shap_values.values.shape[0] and e_idx < shap_values.values.shape[2]:
                                    values = shap_values.values[i, :, e_idx]
                                else:
                                    f.write("    Shape mismatch - cannot extract values\n")
                                    continue
                            elif len(shap_values.values.shape) == 2:
                 
                                if i < shap_values.values.shape[0]:
                                    values = shap_values.values[i, :]
                                else:
                                    f.write("    Shape mismatch - cannot extract values\n")
                                    continue
                            else:
                                f.write("    Unexpected SHAP values shape\n")
                                continue
          
                            if i < len(shap_values.data) and isinstance(shap_values.data[i], (list, np.ndarray)):
                                tokens = shap_values.data[i]
                                
                        
                                min_len = min(len(values), len(tokens))
                                
                                token_values = []
                                for j in range(min_len):
                                    token = tokens[j]
                                    value = values[j]
                                    if token is not None and str(token).strip() != '':
                                        token_values.append((str(token), float(value)))
                            
                                if token_values:
                                    sorted_tokens = sorted(token_values, key=lambda x: abs(x[1]), reverse=True)
                                    f.write("    Most important tokens:\n")
                                    for token, value in sorted_tokens[:10]:
                                        direction = "positive" if value > 0 else "negative"
                                        f.write(f"      {token}: {value:.4f} ({direction} impact)\n")
                                else:
                                    f.write("    No token values available\n")
                            else:
                                f.write("    Cannot access token data\n")
                    except Exception as e:
                        f.write(f"  Error processing token importance: {str(e)}\n")
                else:
                    f.write("  SHAP values object doesn't have the expected structure\n")
                
                f.write("\n" + "-"*50 + "\n\n")
    except Exception as e:
        logger.error(f"Error in generate_token_importance_summary: {str(e)}")
        traceback.print_exc()

def analyze_zero_shot_model(model, tokenizer, model_name, texts, emotion_labels, language, device, output_dir):
    """Analyze zero-shot model performance with SHAP"""
    logger.info(f"Analyzing {model_name} on {language} (zero-shot)")
    

    model_output_dir = os.path.join(output_dir, f"{model_name}_{language}")
    os.makedirs(model_output_dir, exist_ok=True)
    
  
    explainer = TransformerExplainer(model, tokenizer, device, model_name)
    
    predictions = explainer.predict(texts)
    
    generate_basic_shap_plots(model_name, language, texts, predictions, emotion_labels, model_output_dir)
   
    try:
 
        shap_values = safe_shap_explainer(explainer, texts)
        
        if shap_values is not None:

            generate_token_importance_summary(
                shap_values, texts, emotion_labels, model_name, language, model_output_dir
            )
          
            try:

                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values.values,
                    feature_names=emotion_labels,
                    show=False
                )
                plt.title(f"SHAP Summary Plot - {model_name} on {language}")
                plt.tight_layout()
                plt.savefig(os.path.join(model_output_dir, "shap_summary.png"))
                plt.close()
            except Exception as plot_error:
                logger.warning(f"Could not create standard SHAP summary plot: {plot_error}")

                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "SHAP summary plot generation failed.\nSee token importance file for details.",
                        ha='center', va='center', fontsize=12)
                plt.axis('off')
                plt.savefig(os.path.join(model_output_dir, "shap_summary_error.png"))
                plt.close()
    except Exception as e:
        logger.error(f"Error in SHAP analysis for {model_name} on {language}: {str(e)}")
        traceback.print_exc()
   
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"SHAP analysis failed: {str(e)}",
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(model_output_dir, "shap_error.png"))
        plt.close()

def run_zero_shot_shap_analysis():
    """Main function for zero-shot SHAP analysis"""
  
    set_seed(42)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zero_shot_shap_analysis")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to {output_dir}")
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved-models")
    
    zulu_texts, emotion_labels = load_zulu_dataset(num_examples=3)
    pidgin_texts, _ = load_pidgin_dataset(num_examples=3)
    
    logger.info("Loading models and tokenizers...")
    models = load_models_and_tokenizers(models_dir, emotion_labels)
   
    if models['mb_model'] is not None:
        analyze_zero_shot_model(
            models['mb_model'], models['mb_tokenizer'], "mBERT_ZeroShot",
            zulu_texts, emotion_labels, "Zulu", device, output_dir
        )
    else:
        logger.error("Skipping mBERT on Zulu analysis - model not available")
    
    if models['af_model'] is not None:
        analyze_zero_shot_model(
            models['af_model'], models['af_tokenizer'], "AfriBERTa_ZeroShot",
            zulu_texts, emotion_labels, "Zulu", device, output_dir
        )
    else:
        logger.error("Skipping AfriBERTa on Zulu analysis - model not available")
    
    if models['mb_model'] is not None:
        analyze_zero_shot_model(
            models['mb_model'], models['mb_tokenizer'], "mBERT_ZeroShot",
            pidgin_texts, emotion_labels, "PCM", device, output_dir
        )
    else:
        logger.error("Skipping mBERT on Pidgin analysis - model not available")
    
    if models['af_model'] is not None:
        analyze_zero_shot_model(
            models['af_model'], models['af_tokenizer'], "AfriBERTa_ZeroShot",
            pidgin_texts, emotion_labels, "PCM", device, output_dir
        )
    else:
        logger.error("Skipping AfriBERTa on Pidgin analysis - model not available")
    
    logger.info("Zero-shot SHAP analysis completed!")

if __name__ == "__main__":
    run_zero_shot_shap_analysis()
