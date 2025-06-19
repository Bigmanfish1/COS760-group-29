"""
Analyze errors and misclassifications in fine-tuned mBERT and AfriBERTa models
for emotion classification in African languages (Zulu and Nigerian Pidgin English).
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer
)
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset, clean_text, set_seed
from results.evaluate_fine_tuned import load_zulu_dataset, load_pcm_dataset, tokenize_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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
        problem_type="multi_label_classification"
    )
    
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

def get_model_predictions(model, dataloader, device):
    """Get model predictions for a dataset"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_texts = []
    all_logits = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            
            predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits
    }

def analyze_misclassifications(dataset, predictions, labels, emotion_labels, language, model_name, output_dir):
    """Analyze misclassifications and find common patterns"""
    logger.info(f"Analyzing misclassifications for {model_name} on {language}...")
    

    model_output_dir = os.path.join(output_dir, f"{model_name}_{language}")
    os.makedirs(model_output_dir, exist_ok=True)
 
    misclassifications = []
    texts = dataset['test']['text']
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
  
        if not np.array_equal(pred, label):
 
            errors = []
            for j, (p, l) in enumerate(zip(pred, label)):
                if p != l:
    
                    error_type = "False Positive" if p == 1 and l == 0 else "False Negative"
                    errors.append({
                        'emotion': emotion_labels[j],
                        'error_type': error_type,
                        'predicted': bool(p),
                        'actual': bool(l)
                    })
            
            misclassifications.append({
                'index': i,
                'text': texts[i],
                'errors': errors,
                'predictions': {emotion_labels[j]: bool(p) for j, p in enumerate(pred)},
                'labels': {emotion_labels[j]: bool(l) for j, l in enumerate(label)}
            })
    

    misclass_rows = []
    for m in misclassifications:
        for error in m['errors']:
            misclass_rows.append({
                'text': m['text'],
                'emotion': error['emotion'],
                'error_type': error['error_type'],
                'predicted': error['predicted'],                'actual': error['actual']
            })
    
    misclass_df = pd.DataFrame(misclass_rows)
    misclass_file = os.path.join(model_output_dir, "misclassifications.csv")
    misclass_df.to_csv(misclass_file, index=False, encoding='utf-8')
    logger.info(f"Saved misclassifications to {misclass_file}")

    error_counts = misclass_df['emotion'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=error_counts.index, y=error_counts.values)
    plt.title(f'Misclassifications by Emotion - {model_name} on {language}')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Misclassifications')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, "error_by_emotion.png"))
   
    error_type_counts = misclass_df['error_type'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=error_type_counts.index, y=error_type_counts.values)
    plt.title(f'Error Types - {model_name} on {language}')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, "error_types.png"))

    for i, emotion in enumerate(emotion_labels):
        emotion_preds = [p[i] for p in predictions]
        emotion_labels_true = [l[i] for l in labels]
        
        cm = confusion_matrix(emotion_labels_true, emotion_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {emotion} ({model_name} on {language})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, f"confusion_matrix_{emotion}.png"))
    
    for emotion in emotion_labels:
    
        fp_examples = misclass_df[(misclass_df['emotion'] == emotion) & 
                                 (misclass_df['error_type'] == "False Positive")][:10]
        
   
        fn_examples = misclass_df[(misclass_df['emotion'] == emotion) & 
                                 (misclass_df['error_type'] == "False Negative")][:10]
        
       
        with open(os.path.join(model_output_dir, f"{emotion}_examples.txt"), 'w', encoding='utf-8') as f:
            f.write(f"False Positive Examples for {emotion}:\n")
            f.write("-" * 50 + "\n")
            for _, row in fp_examples.iterrows():
                f.write(f"Text: {row['text']}\n")
                f.write(f"Model predicted {emotion}, but it was not labeled as such.\n")
                f.write("-" * 50 + "\n")
            
            f.write(f"\nFalse Negative Examples for {emotion}:\n")
            f.write("-" * 50 + "\n")
            for _, row in fn_examples.iterrows():
                f.write(f"Text: {row['text']}\n")
                f.write(f"Model did not predict {emotion}, but it was labeled as such.\n")
                f.write("-" * 50 + "\n")
    
    logger.info(f"Completed error analysis for {model_name} on {language}")
    return misclass_df

def main():
    """Main function for error analysis"""

    set_seed()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    

    zulu_dataset, emotion_labels = load_zulu_dataset()
    pcm_dataset, _ = load_pcm_dataset()
  
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved-models")
    
    models = load_models_and_tokenizers(models_dir, emotion_labels)
    
    analysis_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "error_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    zulu_mb_dataloader = tokenize_dataset(zulu_dataset, models['mb_tokenizer'])
    zulu_af_dataloader = tokenize_dataset(zulu_dataset, models['af_tokenizer'])
    
   
    mb_zulu_results = get_model_predictions(models['mb_model'], zulu_mb_dataloader, device)
    af_zulu_results = get_model_predictions(models['af_model'], zulu_af_dataloader, device)
    
    
    mb_zulu_misclass = analyze_misclassifications(
        zulu_dataset,
        mb_zulu_results['predictions'],
        mb_zulu_results['labels'],
        emotion_labels,
        "Zulu",
        "mBERT",
        analysis_output_dir
    )
    
    af_zulu_misclass = analyze_misclassifications(
        zulu_dataset,
        af_zulu_results['predictions'],
        af_zulu_results['labels'],
        emotion_labels,
        "Zulu",
        "AfriBERTa",
        analysis_output_dir
    )
    
    
    pcm_mb_dataloader = tokenize_dataset(pcm_dataset, models['mb_tokenizer'])
    pcm_af_dataloader = tokenize_dataset(pcm_dataset, models['af_tokenizer'])
   
    mb_pcm_results = get_model_predictions(models['mb_model'], pcm_mb_dataloader, device)
    af_pcm_results = get_model_predictions(models['af_model'], pcm_af_dataloader, device)
   
    mb_pcm_misclass = analyze_misclassifications(
        pcm_dataset,
        mb_pcm_results['predictions'],
        mb_pcm_results['labels'],
        emotion_labels,
        "PCM",
        "mBERT",
        analysis_output_dir
    )
    
    af_pcm_misclass = analyze_misclassifications(
        pcm_dataset,
        af_pcm_results['predictions'],
        af_pcm_results['labels'],
        emotion_labels,
        "PCM",
        "AfriBERTa",
        analysis_output_dir
    )
    
    logger.info("Error analysis completed.")

if __name__ == "__main__":
    main()
