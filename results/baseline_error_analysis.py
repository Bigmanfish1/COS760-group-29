"""
Analyze errors and misclassifications in baseline (non-fine-tuned) mBERT and AfriBERTa models
for emotion classification in African languages (Zulu and Nigerian Pidgin English).
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"baseline_error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """Dataset class for emotion classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if not text or len(text.strip()) == 0:
            text = "empty text"
            
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def load_baseline_models():
    """Load baseline (non-fine-tuned) models directly from Hugging Face"""
    models = {}
    tokenizers = {}
    logger.info("Loading baseline mBERT model...")
    try:
        tokenizers['mbert'] = AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased",
            local_files_only=False,
            use_fast=True
        )
        models['mbert'] = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", 
            num_labels=6,
            problem_type="multi_label_classification",
            local_files_only=False
        )
        logger.info("Successfully loaded baseline mBERT model")
    except Exception as e:
        logger.error(f"Failed to load mBERT model: {e}")
        logger.error(f"Error details: {str(e)}")
    
    logger.info("Loading baseline AfriBERTa model...")
    try:

        tokenizers['afriberta'] = AutoTokenizer.from_pretrained(
            "castorini/afriberta_large",
            local_files_only=False,
            use_fast=True  
        )
        models['afriberta'] = AutoModelForSequenceClassification.from_pretrained(
            "castorini/afriberta_large", 
            num_labels=6,
            problem_type="multi_label_classification",
            local_files_only=False
        )
        logger.info("Successfully loaded baseline AfriBERTa model")
    except Exception as e:
        logger.error(f"Failed to load AfriBERTa model: {e}")
        logger.error(f"Error details: {str(e)}")
    
    return models, tokenizers

def evaluate_model(model, dataloader, device):
    """Evaluate the model and return metrics and predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                sigmoid_preds = torch.sigmoid(logits).cpu().numpy()
                
                all_preds.extend(sigmoid_preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                continue
    
    if not all_preds:
        raise ValueError("No predictions were made. Check if the model is compatible with the data.")
    
    logger.info(f"Evaluated {len(all_preds)} samples")
    return all_preds, all_labels

def error_analysis(preds, labels, texts, emotion_labels, model_name, language, output_dir):
    """Perform detailed error analysis for the model predictions"""
    os.makedirs(output_dir, exist_ok=True)


    preds = np.array(preds)
    labels = np.array(labels)
    
 
    full_results = pd.DataFrame({
        'text': texts,
    })

    for i, emotion in enumerate(emotion_labels):
        full_results[f'true_{emotion}'] = labels[:, i]

    for i, emotion in enumerate(emotion_labels):
        full_results[f'pred_{emotion}'] = preds[:, i]
    

    dominant_true_indices = np.argmax(labels, axis=1)
    dominant_pred_indices = np.argmax(preds, axis=1)
    
   
    text_labels = [emotion_labels[idx] for idx in dominant_true_indices]
    pred_labels = [emotion_labels[idx] for idx in dominant_pred_indices]
    

    df = pd.DataFrame({
        'text': texts,
        'true_label': text_labels,
        'predicted_label': pred_labels,
        'correct': [p == t for p, t in zip(dominant_pred_indices, dominant_true_indices)]
    })
    

    full_results.to_csv(os.path.join(output_dir, f"{model_name}_{language}_full_predictions.csv"), index=False)
    df.to_csv(os.path.join(output_dir, f"{model_name}_{language}_predictions.csv"), index=False)
    logger.info(f"Saved predictions to {model_name}_{language}_predictions.csv")

    report = classification_report(dominant_true_indices, dominant_pred_indices, 
                                  target_names=emotion_labels, output_dict=True,
                                  zero_division=0) 
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{model_name}_{language}_report.csv"))

    cm = confusion_matrix(dominant_true_indices, dominant_pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name} on {language}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(emotion_labels))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = []
        for j, emotion in enumerate(emotion_labels):

            if emotion in report:
                values.append(report[emotion][metric])
            else:
                values.append(0) 
        plt.bar(x + (i - 1) * width, values, width, label=metric)
    
    plt.xlabel('Emotions')
    plt.ylabel('Score')
    plt.title(f'Performance by Emotion - {model_name} on {language}')
    plt.xticks(x, emotion_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_emotion_performance.png"))
    plt.close()

    misclassified = df[~df['correct']]
    error_patterns = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
    error_patterns = error_patterns.sort_values('count', ascending=False)
    
    plt.figure(figsize=(10, 8))
    pivot = pd.pivot_table(error_patterns, values='count', index='true_label', columns='predicted_label', fill_value=0)
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Reds')
    plt.title(f'Common Misclassification Patterns - {model_name} on {language}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_{language}_misclassification_patterns.png"))
    plt.close()

    top_errors = []
    for true_emotion in emotion_labels:
        for pred_emotion in emotion_labels:
            if true_emotion != pred_emotion:
                subset = df[(df['true_label'] == true_emotion) & (df['predicted_label'] == pred_emotion)]
                if len(subset) > 0:
                    top_errors.append(subset.iloc[0:min(3, len(subset))])
    
    if top_errors:
        top_error_df = pd.concat(top_errors)
        top_error_df.to_csv(os.path.join(output_dir, f"{model_name}_{language}_top_errors.csv"), index=False)
    
    return report

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    

    logger.info("Loading datasets...")
    dataset_dict = load_and_preprocess_data()
    
   
    emotion_labels = dataset_dict['emotion_labels']
    logger.info(f"Emotion labels: {emotion_labels}")
  
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "error_analysis", "baseline")
    os.makedirs(output_dir, exist_ok=True)
    
    models, tokenizers = load_baseline_models()
    
    for model_name, model in models.items():
        models[model_name] = model.to(device)
    
    languages = ['zul', 'pcm']
    batch_size = 16
    
    for lang in languages:
        logger.info(f"Processing {lang} dataset...")
        
        dataset_name = 'zulu_dataset' if lang == 'zul' else 'pidgin_dataset'
        test_data = dataset_dict[dataset_name]['test']
        test_texts = test_data['text']
        
        test_labels = test_data['labels']
        logger.info(f"Loaded {len(test_texts)} test samples for {lang}")
        
        lang_output_dir = os.path.join(output_dir, lang)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        for model_name, model in models.items():
            logger.info(f"Evaluating baseline {model_name} model on {lang}...")
            
            try:
   
                tokenizer = tokenizers[model_name]
                test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                predictions, labels = evaluate_model(model, test_dataloader, device)
     
                report = error_analysis(
                    predictions, 
                    labels, 
                    test_texts, 
                    emotion_labels, 
                    model_name, 
                    lang, 
                    lang_output_dir
                )

                f1_weighted = report['weighted avg']['f1-score']
                logger.info(f"{model_name} on {lang} - Weighted F1: {f1_weighted:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {lang}: {e}")
                continue

if __name__ == "__main__":
    main()
