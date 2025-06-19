"""
Use SHAP (SHapley Additive exPlanations) to explain predictions of fine-tuned
mBERT and AfriBERTa models for emotion classification in African languages.
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


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset, clean_text, set_seed
from results.evaluate_fine_tuned import load_zulu_dataset, load_pcm_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_models_and_tokenizers(models_dir, emotion_labels):
    """Load the fine-tuned models and tokenizers"""
    # mBERT
    logger.info("Loading fine-tuned mBERT model...")
    mb_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    mb_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(models_dir, "mbert_english"),
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    )
    
    # AfriBERTa
    logger.info("Loading fine-tuned AfriBERTa model...")
    af_tokenizer = XLMRobertaTokenizer.from_pretrained("castorini/afriberta_base")
    af_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(models_dir, "afriberta_english"),
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    )
    
    return {
        'mb_model': mb_model,
        'mb_tokenizer': mb_tokenizer,
        'af_model': af_model,
        'af_tokenizer': af_tokenizer
    }

class TransformerExplainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, texts):
        """Predict for a batch of texts and return logits"""
     
        if not isinstance(texts, list):
            if isinstance(texts, str):
                texts = [texts]  
            else:
                texts = list(map(str, texts)) 
        else:
            
            texts = [str(text) for text in texts]
        
        
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
    
    def predict_for_shap(self, texts):
        """A wrapper for predict that returns a numpy array"""
        return self.predict(texts)

def generate_shap_visualizations(explainer, model_type, texts, emotion_labels, language, output_dir):
    """Generate SHAP visualizations for a set of texts"""
    logger.info(f"Generating SHAP visualizations for {model_type} on {language}...")
    
  
    model_output_dir = os.path.join(output_dir, f"{model_type}_{language}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    try:
      
        masker = shap.maskers.Text(explainer.tokenizer)
        
        
        shap_explainer = shap.Explainer(explainer.predict_for_shap, masker=masker, output_names=emotion_labels)
        
        
        with np.errstate(invalid='ignore'):
            shap_values = shap_explainer(texts)
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {str(e)}")
    
        try:
            logger.info("Trying alternative SHAP calculation approach...")
            
            if len(texts) > 5:
                texts = texts[:5]
                logger.info(f"Reduced to {len(texts)} examples for SHAP analysis")
            
            masker = shap.maskers.Text(explainer.tokenizer)
            shap_explainer = shap.Explainer(explainer.predict_for_shap, masker=masker, output_names=emotion_labels)
            with np.errstate(invalid='ignore'):
                shap_values = shap_explainer(texts)
        except Exception as e2:
            logger.error(f"Alternative SHAP calculation also failed: {str(e2)}")
            return 

    for i, emotion in enumerate(emotion_labels):
        try:
     
            plt.figure(figsize=(12, len(texts) * 0.8))
            
        
            try:
              
                if hasattr(shap_values, 'values') and isinstance(shap_values.values, np.ndarray):
                 
                    shape = shap_values.values.shape
                    
                    
                    if len(shape) == 3 and i < shape[2]:
                       
                        shap_values_for_emotion = shap_values.values[:, :, i].copy()
                        
                        temp_values = shap.Explanation(
                            values=shap_values_for_emotion, 
                            data=shap_values.data if hasattr(shap_values, 'data') else None,
                            feature_names=shap_values.feature_names if hasattr(shap_values, 'feature_names') else None
                        )
                        
                        shap.plots.text(temp_values, display=False)
                    elif len(shape) == 2:
                     
                        logger.info(f"Using 2D format for visualization with shape {shape}")
                        shap.plots.text(shap_values, display=False)
                    else:
                        logger.warning(f"Unexpected SHAP values shape: {shape}")
                        shap.plots.text(shap_values, display=False)
                else:
                    
                    shap.plots.text(shap_values, display=False)
            except Exception as text_plot_error:
                logger.warning(f"Could not create standard text plot: {str(text_plot_error)}")
              
                try:
                    shap.plots.force(shap_values[0], matplotlib=True)
                except Exception as force_plot_error:
                    logger.warning(f"Force plot also failed: {str(force_plot_error)}")
                    
                    plt.text(0.5, 0.5, f"Could not visualize SHAP values: {str(text_plot_error)}", 
                            ha='center', va='center', fontsize=12, wrap=True)
                    plt.axis('off')
                
            plt.title(f"SHAP values for {emotion} - {model_type} on {language}")
            plt.tight_layout()
            plt.savefig(os.path.join(model_output_dir, f"shap_{emotion}.png"), bbox_inches='tight')
            plt.close()
            
          
            with open(os.path.join(model_output_dir, f"shap_values_{emotion}.txt"), 'w', encoding='utf-8') as f:
                for j, text in enumerate(texts):
                    f.write(f"Text: {text}\n")
               
                    token_explanations = []
                   
                    try:
                        if hasattr(shap_values, 'values') and hasattr(shap_values, 'data'):
                            try:
                               
                                if isinstance(shap_values.values, np.ndarray):
                                    if len(shap_values.values.shape) == 3 and i < shap_values.values.shape[2]:
                                      
                                        if j < shap_values.values.shape[0]:
                                            values = shap_values.values[j, :, i]
                                        else:
                                            f.write(f"Warning: Example index {j} out of bounds for shape {shap_values.values.shape}\n")
                                            continue
                                    elif len(shap_values.values.shape) == 2:
                                        
                                        if j < shap_values.values.shape[0]:
                                            values = shap_values.values[j, :]
                                        else:
                                            f.write(f"Warning: Example index {j} out of bounds for shape {shap_values.values.shape}\n")
                                            continue
                                    else:
                                       
                                        if j < len(shap_values.values):
                                            values = shap_values.values[j]
                                        else:
                                            f.write(f"Warning: Example index {j} out of bounds for shape {shap_values.values.shape}\n")
                                            continue
                                else:
                                 
                                    f.write("SHAP values not in numpy array format\n")
                                    continue
                                
                                
                                if j < len(shap_values.data) and isinstance(shap_values.data[j], (list, np.ndarray)):
       
                                    min_len = min(len(values), len(shap_values.data[j]))
                                    for k in range(min_len):
                                        token = shap_values.data[j][k]
                                        value = values[k]
                                        if token is not None and str(token).strip() != '':  
                                            token_explanations.append((str(token), float(value)))
                                else:
                                    f.write(f"Warning: Data index {j} out of bounds or not list-like\n")
                            except Exception as e:
                                f.write(f"Error processing SHAP values for example {j}: {str(e)}\n")
                        else:
                          
                            f.write("Note: SHAP values structure doesn't contain token-level data\n")
                    except (IndexError, AttributeError) as e:
                        f.write(f"Error accessing SHAP values: {str(e)}\n")
                    
           
                    if token_explanations:
                        sorted_explanations = sorted(token_explanations, key=lambda x: abs(float(x[1])), reverse=True)
                    else:
                        sorted_explanations = []
                    
                    f.write("Most important tokens:\n")
                    if sorted_explanations:
                       
                        for token, value in sorted_explanations[:min(10, len(sorted_explanations))]:
                            try:
                                impact = "positive" if float(value) > 0 else "negative"
                                f.write(f"- {token}: {float(value):.4f} ({impact} impact)\n")
                            except (ValueError, TypeError) as e:
                                f.write(f"- Error processing token '{token}': {str(e)}\n")
                    else:
                        f.write("No token explanations available\n")
                    
                    f.write("-" * 50 + "\n")
        except Exception as e:
            logger.error(f"Error creating SHAP visualization for {emotion}: {str(e)}")
            continue
  
    try:
        plt.figure(figsize=(12, 8))
        
        try:
            
            if hasattr(shap_values, 'values') and isinstance(shap_values.values, np.ndarray):
                
                if len(shap_values.values.shape) == 3:  
                   
                    logger.info(f"Using multi-class format with shape {shap_values.values.shape}")
                    
                    for cls_idx, emotion_name in enumerate(emotion_labels):
                        if cls_idx < shap_values.values.shape[2]:
                            
                            cls_values = shap_values.values[:, :, cls_idx].copy()
                            
                            if len(cls_values.shape) == 1:
                                cls_values = cls_values.reshape(-1, 1)
                                
                            plt.figure(figsize=(10, 8))
                            
                            features = None
                            if hasattr(shap_values, 'data') and shap_values.data is not None:
                                features = shap_values.data
                            
                            feature_names = None
                            if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
                                feature_names = shap_values.feature_names
                                
                            shap.summary_plot(
                                cls_values,
                                features=features,
                                feature_names=feature_names,
                                show=False,
                                plot_type="bar"
                            )
                            plt.title(f"SHAP Summary for {emotion_name} - {model_type} on {language}")
                            plt.tight_layout()
                            plt.savefig(os.path.join(model_output_dir, f"shap_summary_{emotion_name}.png"), 
                                        bbox_inches='tight')
                            plt.close()
                    
                   
                    mean_abs_values = np.mean(np.abs(shap_values.values), axis=2).copy()
                    
                    if len(mean_abs_values.shape) == 1:
                        mean_abs_values = mean_abs_values.reshape(-1, 1)
                        
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        mean_abs_values,
                        features=shap_values.data if hasattr(shap_values, 'data') else None,
                        feature_names=shap_values.feature_names if hasattr(shap_values, 'feature_names') else None,
                        show=False,
                        plot_type="bar"
                    )
                    plt.title(f"SHAP Summary Plot (All Emotions) - {model_type} on {language}")
                else:
                  
                    try:
                       
                        if len(shap_values.values.shape) == 1:
                          
                            num_samples = len(texts)
                            flat_values = shap_values.values.reshape(num_samples, -1)
                        else:
                           
                            flat_values = shap_values.values.copy()
                            
                        logger.info(f"Using flattened format with shape {flat_values.shape}")
                        
                
                        if len(flat_values.shape) < 2:
                            raise ValueError(f"Cannot create summary plot with shape {flat_values.shape}")
                            
                        shap.summary_plot(
                            flat_values,
                            features=shap_values.data if hasattr(shap_values, 'data') else None,
                            feature_names=shap_values.feature_names if hasattr(shap_values, 'feature_names') else None,
                            show=False,
                            plot_type="bar"
                        )
                    except Exception as reshape_error:
                        logger.error(f"Error reshaping for summary plot: {str(reshape_error)}")
            
                        plt.text(0.5, 0.5, f"Could not create summary plot: {str(reshape_error)}", 
                                ha='center', va='center', fontsize=12)
                        plt.axis('off')
            else:
       
                logger.info("Using direct summary plot")
                shap.summary_plot(shap_values, show=False, plot_type="bar")
        except Exception as summary_error:
            logger.warning(f"Standard summary plot failed: {str(summary_error)}")
      
            try:
                if hasattr(shap_values, 'values') and isinstance(shap_values.values, np.ndarray):
                  
                    if len(shap_values.values.shape) == 3:
                        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))
                    else:
                        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
                    
                    feature_names = None
                    if hasattr(shap_values, 'feature_names'):
                        feature_names = shap_values.feature_names
                    if feature_names is None or len(feature_names) != len(mean_abs_shap):
                        feature_names = [f"Feature {i}" for i in range(len(mean_abs_shap))]
                    
               
                    sorted_idx = np.argsort(mean_abs_shap)[::-1][:20]  
                    sorted_names = [feature_names[i] for i in sorted_idx]
                    sorted_values = mean_abs_shap[sorted_idx]
                  
                    plt.figure(figsize=(12, 8))
                    plt.barh(range(len(sorted_idx)), sorted_values)
                    plt.yticks(range(len(sorted_idx)), sorted_names)
                    plt.xlabel('Mean |SHAP value|')
                    plt.title(f"Feature Importance - {model_type} on {language}")
                else:
                    plt.text(0.5, 0.5, f"Could not create summary plot: {str(summary_error)}", 
                            ha='center', va='center', fontsize=12, wrap=True)
                    plt.axis('off')
            except Exception as backup_error:
                logger.error(f"Backup summary visualization also failed: {str(backup_error)}")
                plt.text(0.5, 0.5, f"Could not create any visualization: {str(backup_error)}", 
                        ha='center', va='center', fontsize=12, wrap=True)
                plt.axis('off')
            
        plt.title(f"SHAP Summary Plot - {model_type} on {language}")
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, "shap_summary.png"), bbox_inches='tight')
        plt.close()
        logger.info(f"Summary SHAP visualization for {model_type} on {language} saved")
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {str(e)}")
    
    logger.info(f"SHAP visualizations for {model_type} on {language} saved to {model_output_dir}")

def main():
    """Main function for SHAP analysis"""

    set_seed()
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    zulu_dataset, emotion_labels = load_zulu_dataset()
    pcm_dataset, _ = load_pcm_dataset()
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved-models")
    
    models = load_models_and_tokenizers(models_dir, emotion_labels)
    
    shap_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shap_analysis")
    os.makedirs(shap_output_dir, exist_ok=True)
    
    
    num_examples = 5  
    logger.info(f"Using {num_examples} examples for SHAP analysis")
    
   
    zulu_texts = [str(zulu_dataset['test'][i]['text']).strip() for i in range(num_examples)]
    pcm_texts = [str(pcm_dataset['test'][i]['text']).strip() for i in range(num_examples)]
   
    mb_explainer = TransformerExplainer(models['mb_model'], models['mb_tokenizer'], device)
    af_explainer = TransformerExplainer(models['af_model'], models['af_tokenizer'], device)
    
    # Generate SHAP visualizations for mBERT on Zulu
    generate_shap_visualizations(
        mb_explainer,
        "mBERT",
        zulu_texts,
        emotion_labels,
        "Zulu",
        shap_output_dir
    )
    
    # Generate SHAP visualizations for AfriBERTa on Zulu
    generate_shap_visualizations(
        af_explainer,
        "AfriBERTa",
        zulu_texts,
        emotion_labels,
        "Zulu",
        shap_output_dir
    )
    
    # Generate SHAP visualizations for mBERT on Nigerian Pidgin English
    generate_shap_visualizations(
        mb_explainer,
        "mBERT",
        pcm_texts,
        emotion_labels,
        "PCM",
        shap_output_dir
    )
    
    # Generate SHAP visualizations for AfriBERTa on Nigerian Pidgin English
    generate_shap_visualizations(
        af_explainer,
        "AfriBERTa",
        pcm_texts,
        emotion_labels,
        "PCM",
        shap_output_dir
    )
    
    logger.info("SHAP analysis completed.")

if __name__ == "__main__":
    main()
