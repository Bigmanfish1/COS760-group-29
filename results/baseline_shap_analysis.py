"""
Use SHAP (SHapley Additive exPlanations) to explain predictions of baseline (non-fine-tuned) 
mBERT and AfriBERTa models for emotion classification in African languages.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification
)
import logging
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_and_preprocess_data, clean_text, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"baseline_shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_baseline_models():
    """Load baseline (non-fine-tuned) models directly from Hugging Face"""
    models = {}
    tokenizers = {}
    
    logger.info("Loading baseline mBERT model...")
    try:
        tokenizers['mbert'] = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        models['mbert'] = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", 
            num_labels=6,
            problem_type="multi_label_classification"
        )
        logger.info("Successfully loaded baseline mBERT model")
    except Exception as e:
        logger.error(f"Failed to load mBERT model: {e}")
        logger.error(f"Error details: {str(e)}")
    
    logger.info("Loading baseline AfriBERTa model...")
    try:
        tokenizers['afriberta'] = XLMRobertaTokenizer.from_pretrained("castorini/afriberta_large")
        models['afriberta'] = XLMRobertaForSequenceClassification.from_pretrained(
            "castorini/afriberta_large", 
            num_labels=6,
            problem_type="multi_label_classification"
        )
        logger.info("Successfully loaded baseline AfriBERTa model")
    except Exception as e:
        logger.error(f"Failed to load AfriBERTa model: {e}")
        logger.error(f"Error details: {str(e)}")
    
    return models, tokenizers

def model_predict(inputs, model, device):
    """Model prediction function for SHAP"""
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return torch.sigmoid(outputs.logits).cpu().numpy()

def create_shap_explainer(model, tokenizer, device, model_type='bert'):
    """Create a SHAP explainer for the model"""
    logger.info(f"Creating SHAP explainer for model type: {model_type}")

    def prepare_and_predict(text_batch):

        if isinstance(text_batch, str):
            text_batch = [text_batch]
        elif not isinstance(text_batch, list):
            try:
                text_batch = list(text_batch)
            except:
                text_batch = [str(text_batch)]
   
        text_batch = [str(text) for text in text_batch]
            
        try:
      
            batch_encoding = tokenizer(
                text_batch, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            return model_predict(batch_encoding, model, device)
        except Exception as e:
            logger.warning(f"Error in prepare_and_predict: {e}")
       
            return np.zeros((len(text_batch), 6))
    

    for attempt in range(1, 4):
        try:
            logger.info(f"Attempt {attempt} to create SHAP explainer")
            if attempt == 1:
  
                explainer = shap.Explainer(prepare_and_predict)
            elif attempt == 2:

                try:

                    masker = shap.maskers.Text(tokenizer)
                    explainer = shap.Explainer(prepare_and_predict, masker=masker)
                except:
    
                    explainer = shap.explainers.Partition(prepare_and_predict)
            else:

                explainer = shap.explainers.Sampling(prepare_and_predict)
                
            logger.info(f"Successfully created SHAP explainer on attempt {attempt}")
            return explainer
        except Exception as e:
            logger.error(f"Error creating SHAP explainer (attempt {attempt}): {e}")
    
    logger.error("All attempts to create SHAP explainer failed")
    return None

def generate_shap_values(explainer, texts, emotion_labels, model_name, language, output_dir):
    """Generate SHAP values and visualizations for model predictions"""
    if not explainer:
        logger.error(f"Cannot generate SHAP values - no explainer available")
        return None
    

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Computing SHAP values for {model_name} on {language}...")

        sample_size = min(100, len(texts))
        sample_texts = texts[:sample_size]
        shap_values = explainer(sample_texts)
        

        np.save(os.path.join(output_dir, f"{model_name}_{language}_shap_values.npy"), shap_values.values)
        

        sample_indices = list(range(min(10, sample_size)))
        pred_df = pd.DataFrame()
        
        for i in sample_indices:
            row = {}
            row['text'] = sample_texts[i]
            for emotion_idx, emotion in enumerate(emotion_labels):
                row[emotion] = shap_values.values[i, :, emotion_idx].sum()
            pred_df = pd.concat([pred_df, pd.DataFrame([row])], ignore_index=True)
        
        pred_df.to_csv(os.path.join(output_dir, f"{model_name}_{language}_predictions.csv"), index=False)
        

        for i in sample_indices:
            plt.figure(figsize=(10, 6))
            scores = [shap_values.values[i, :, emotion_idx].sum() for emotion_idx in range(len(emotion_labels))]
            sns.barplot(x=emotion_labels, y=scores)
            plt.title(f'Prediction Scores - {model_name} on {language} - Sample {i+1}')
            plt.xlabel('Emotion')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_prediction_scores_sample{i+1}.png"))
            plt.close()

        try:

            plt.figure(figsize=(12, 8))
            try:
                shap.plots.bar(shap_values, show=False)
                plt.title(f'SHAP Feature Importance - {model_name} on {language}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_shap_feature_importance.png"))
            except Exception as e:
                logger.warning(f"Could not create bar plot: {e}")

                plt.title(f'SHAP Analysis Failed - {model_name} on {language}')
                plt.text(0.5, 0.5, f"Bar plot generation failed: {str(e)[:100]}...", 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_shap_feature_importance_error.png"))
            finally:
                plt.close()
            

            for i in range(min(5, len(texts))):
                try:
                    plt.figure(figsize=(12, 8))

                    if hasattr(shap.plots, 'text'):
                        shap.plots.text(shap_values[i], show=False)
                    else:
   
                        shap_values_for_viz = shap_values.values[i] if hasattr(shap_values, 'values') else shap_values[i]
                        shap.plots.text(shap_values_for_viz)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_shap_text_sample{i+1}.png"))
                except Exception as e:
                    logger.warning(f"Could not create text plot for sample {i+1}: {e}")
                    plt.title(f'SHAP Text Plot Failed - Sample {i+1}')
                    plt.text(0.5, 0.5, f"Text plot generation failed: {str(e)[:100]}...", 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_shap_text_sample{i+1}_error.png"))
                finally:
                    plt.close()
        except Exception as e:
            logger.warning(f"Could not generate standard SHAP plots due to error: {e}")
            logger.info("Generating alternative visualizations...")
            
          
            for i in range(min(5, len(texts))):
                try:

                    token_importance = np.abs(shap_values.values[i]).sum(axis=1)
                    tokens = texts[i].split()[:len(token_importance)]
          
                    plt.figure(figsize=(12, 3))
                    df = pd.DataFrame({'token': tokens, 'importance': token_importance[:len(tokens)]})
                    
                    
                    heatmap_data = df.set_index('token')['importance'].values.reshape(1, -1)
                    ax = sns.heatmap(heatmap_data, annot=df['token'].values.reshape(1, -1), 
                                    fmt='', cmap='viridis', cbar=True)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    plt.title(f'Token Importance - {model_name} on {language} - Sample {i+1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_token_importance_sample{i+1}.png"))
                    plt.close()
           
                    df.to_csv(os.path.join(output_dir, f"{model_name}_{language}_token_importance_sample{i+1}.csv"), index=False)
                except Exception as e:
                    logger.warning(f"Could not generate token heatmap for sample {i+1}: {e}")
        
        return shap_values
    except Exception as e:
        logger.error(f"Error generating SHAP values: {e}")
        return None

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    

    logger.info("Loading datasets...")
    dataset_dict = load_and_preprocess_data()

    emotion_labels = dataset_dict['emotion_labels']
    logger.info(f"Emotion labels: {emotion_labels}")
    

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "shap_analysis", "baseline")
 
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {e}")
      
        output_dir = os.path.join(base_dir, "baseline_shap_results")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using fallback output directory: {output_dir}")
    
 
    models, tokenizers = load_baseline_models()
   
    for model_name, model in models.items():
        models[model_name] = model.to(device)
 
    languages = ['zul', 'pcm']
    
    for lang in languages:
        logger.info(f"Processing {lang} dataset...")
  
        dataset_name = 'zulu_dataset' if lang == 'zul' else 'pidgin_dataset'
        
        try:
            test_data = dataset_dict[dataset_name]['test']
            test_texts = [str(text).strip() for text in test_data['text']]
            logger.info(f"Loaded {len(test_texts)} test samples for {lang}")
        except KeyError as e:
            logger.error(f"KeyError accessing dataset: {e}. Available keys: {dataset_dict.keys()}")
            logger.error(f"Skipping {lang} dataset")
            continue
      
        lang_output_dir = os.path.join(output_dir, lang)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        for model_name, model in models.items():
            logger.info(f"Generating explanations for baseline {model_name} model on {lang}...")
            
            try:
          
                tokenizer = tokenizers[model_name]
                model_type = 'bert' if model_name == 'mbert' else 'roberta'
                explainer = create_shap_explainer(model, tokenizer, device, model_type)

                shap_values = generate_shap_values(
                    explainer, 
                    test_texts,
                    emotion_labels,
                    model_name, 
                    lang, 
                    lang_output_dir
                )
                
                logger.info(f"Completed SHAP analysis for {model_name} on {lang}")
                
            except Exception as e:
                logger.error(f"Error analyzing {model_name} on {lang}: {e}")
                continue

if __name__ == "__main__":
    main()
