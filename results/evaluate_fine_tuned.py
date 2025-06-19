"""
Evaluate fine-tuned mBERT and AfriBERTa models on Zulu and Nigerian Pidgin English test sets
from the BRIGHTER dataset for offensive language detection.
Using English-trained models as language-specific models are not yet available.
"""
import os
import sys
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
import logging
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset, clean_text, set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"fine_tuned_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_zulu_dataset():
    """Load and preprocess the Zulu BRIGHTER dataset"""
    logger.info("Loading Zulu BRIGHTER dataset...")
    brighter_zul = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "zul")
    
    brighter_zul = brighter_zul.map(lambda x: {"text": clean_text(x["text"])})
    
    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    brighter_zul = brighter_zul.map(lambda example: {
        'labels': [example[e] for e in emotion_labels]
    })
    
    return brighter_zul, emotion_labels

def load_pcm_dataset():
    """Load and preprocess the Nigerian Pidgin English (PCM) BRIGHTER dataset"""
    logger.info("Loading Nigerian Pidgin English BRIGHTER dataset...")
    brighter_pcm = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "pcm")
    
    brighter_pcm = brighter_pcm.map(lambda x: {"text": clean_text(x["text"])})
    
    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    brighter_pcm = brighter_pcm.map(lambda example: {
        'labels': [example[e] for e in emotion_labels]
    })
    
    return brighter_pcm, emotion_labels

def tokenize_dataset(dataset, tokenizer, batch_size=8):
    """Tokenize the dataset and create a DataLoader"""
    
    class EmotionDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
            
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
       
            processed_label = []
            for item in label:
                if item is None:
                    processed_label.append(0)
                else:
                    processed_label.append(int(item))
            
       
            label_tensor = torch.tensor(processed_label, dtype=torch.float)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label_tensor
            }
   
    test_dataset = EmotionDataset(
        dataset['test']['text'],
        dataset['test']['labels'],
        tokenizer
    )
    

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    return test_dataloader

def evaluate_model(model, dataloader, device, model_name, emotion_labels):
    """Evaluate a fine-tuned model on a test dataset"""
    logger.info(f"Evaluating {model_name} model...")
    

    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    eval_loss = 0
    
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
            eval_loss += outputs.loss.item()
            
            predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
   
    avg_eval_loss = eval_loss / len(dataloader)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    per_class_f1 = []
    for i in range(len(emotion_labels)):
        class_labels = [label[i] for label in all_labels]
        class_preds = [pred[i] for pred in all_predictions]
        class_f1 = f1_score(class_labels, class_preds, zero_division=0)
        per_class_f1.append((emotion_labels[i], class_f1))
    
    logger.info(f"{model_name} - Test Loss: {avg_eval_loss}, F1 Macro: {f1_macro}, F1 Micro: {f1_micro}, F1 Weighted: {f1_weighted}")
    logger.info(f"{model_name} - Per-class F1 scores:")
    for label, score in per_class_f1:
        logger.info(f"  {label}: {score:.4f}")
    
    logger.info(f"{model_name} - Classification Report:")
    report = classification_report(all_labels, all_predictions, target_names=emotion_labels, zero_division=0)
    logger.info(f"\n{report}")
    
    return {
        'loss': avg_eval_loss,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'per_class_f1': dict(per_class_f1),
        'classification_report': report
    }

def main():
    """Main function to run the evaluation process"""
   
    set_seed()
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved-models")
    
    zulu_dataset, emotion_labels = load_zulu_dataset()
    logger.info(f"Loaded Zulu dataset with test size: {len(zulu_dataset['test'])}")

    logger.info("Loading fine-tuned mBERT model for English...")
    mb_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    mb_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(models_dir, "mbert_english"),
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    )
   
    mb_test_dataloader = tokenize_dataset(zulu_dataset, mb_tokenizer)
    

    mb_results = evaluate_model(
        mb_model,
        mb_test_dataloader,
        device,
        "mBERT (Zulu)",
        emotion_labels
    )

    logger.info("Loading fine-tuned AfriBERTa model for English...")
    af_tokenizer = XLMRobertaTokenizer.from_pretrained("castorini/afriberta_base")
    af_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(models_dir, "afriberta_english"),
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    )
    

    af_test_dataloader = tokenize_dataset(zulu_dataset, af_tokenizer)
 
    af_results = evaluate_model(
        af_model,
        af_test_dataloader,
        device,
        "AfriBERTa (Zulu)",
        emotion_labels
    )    
    pcm_dataset, _ = load_pcm_dataset()
    logger.info(f"Loaded Nigerian Pidgin English dataset with test size: {len(pcm_dataset['test'])}")
    
 
    logger.info("Evaluating mBERT model on Nigerian Pidgin English...")
    mb_pcm_test_dataloader = tokenize_dataset(pcm_dataset, mb_tokenizer)
    
    mb_pcm_results = evaluate_model(
        mb_model,
        mb_pcm_test_dataloader,
        device,
        "mBERT (Nigerian Pidgin English)",
        emotion_labels
    )
    

    logger.info("Evaluating AfriBERTa model on Nigerian Pidgin English...")
    af_pcm_test_dataloader = tokenize_dataset(pcm_dataset, af_tokenizer)
    
    af_pcm_results = evaluate_model(
        af_model,
        af_pcm_test_dataloader,
        device,
        "AfriBERTa (Nigerian Pidgin English)",
        emotion_labels
    )    
    logger.info("\n===== SUMMARY OF RESULTS =====")
    
    logger.info("\nEnglish Models on Zulu Test Set Comparison:")
    logger.info(f"mBERT F1 Macro: {mb_results['f1_macro']:.4f}, F1 Micro: {mb_results['f1_micro']:.4f}")
    logger.info(f"AfriBERTa F1 Macro: {af_results['f1_macro']:.4f}, F1 Micro: {af_results['f1_micro']:.4f}")
    
    logger.info("\nEnglish Models on Nigerian Pidgin English Test Set Comparison:")
    logger.info(f"mBERT F1 Macro: {mb_pcm_results['f1_macro']:.4f}, F1 Micro: {mb_pcm_results['f1_micro']:.4f}")
    logger.info(f"AfriBERTa F1 Macro: {af_pcm_results['f1_macro']:.4f}, F1 Micro: {af_pcm_results['f1_micro']:.4f}")
    
    logger.info("\nNote: Models specifically fine-tuned for Zulu and Nigerian Pidgin English were not found.")
    logger.info("This evaluation used the available English-trained models on both test sets.")
    
    logger.info("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
