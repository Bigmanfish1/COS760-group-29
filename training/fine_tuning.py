"""
Fine-tune mBERT and AfriBERTa models on English BRIGHTER dataset
for zero-shot cross-lingual transfer to African languages.
"""
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset, clean_text, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_english_dataset():
    logger.info("Loading English BRIGHTER dataset...")
    brighter_eng = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng")
    
    brighter_eng = brighter_eng.map(lambda x: {"text": clean_text(x["text"])})
    
    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    brighter_eng = brighter_eng.map(lambda example: {
        'labels': [example[e] for e in emotion_labels]
    })
    
    return brighter_eng, emotion_labels

def tokenize_dataset(dataset, tokenizer, batch_size=8):
    """Tokenize the dataset and create DataLoaders"""
    
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

    train_dataset = EmotionDataset(
        dataset['train']['text'],
        dataset['train']['labels'],
        tokenizer
    )
    
    eval_dataset = EmotionDataset(
        dataset['dev']['text'],
        dataset['dev']['labels'],
        tokenizer
    )
 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size
    )
    
    return train_dataloader, eval_dataloader

def train_model(model, train_dataloader, eval_dataloader, device, model_name, output_dir, 
               num_epochs=3, learning_rate=2e-5):
    """Fine-tune a pre-trained model on the English dataset"""
    

    model = model.to(device)
   
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
 
    logger.info(f"Starting training of {model_name} model...")
    best_f1 = 0.0
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
   
        model.train()
        train_loss = 0
        train_progress = tqdm(train_dataloader, desc=f"Training {model_name}")
        
        for batch in train_progress:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
   
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
       
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_progress.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"{model_name} - Avg training loss: {avg_train_loss}")
        
        model.eval()
        all_predictions = []
        all_labels = []
        eval_loss = 0
        
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {model_name}"):

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
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        logger.info(f"{model_name} - Validation Loss: {avg_eval_loss}, F1 Score: {f1_macro}")
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            logger.info(f"{model_name} - New best model with F1: {best_f1}")
            
            model_save_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_save_dir, exist_ok=True)
            
            model.save_pretrained(model_save_dir)
            logger.info(f"Model saved to {model_save_dir}")
    
    return model

def main():
    """Main function to run the fine-tuning process"""
    
    set_seed()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    english_dataset, emotion_labels = load_english_dataset()
    logger.info(f"Loaded English dataset with splits: {english_dataset.keys()}")
    logger.info(f"Train size: {len(english_dataset['train'])}")
    logger.info(f"Dev size: {len(english_dataset['dev'])}")
    logger.info(f"Test size: {len(english_dataset['test'])}")
    
   
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved-models")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Preparing mBERT model...")
    mb_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    mb_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    )
    
    mb_train_dataloader, mb_eval_dataloader = tokenize_dataset(english_dataset, mb_tokenizer)
   
    mb_model = train_model(
        mb_model, 
        mb_train_dataloader, 
        mb_eval_dataloader, 
        device, 
        "mbert_english", 
        output_dir, 
        num_epochs=3
    )
    
    logger.info("Preparing AfriBERTa model...")
    af_tokenizer = XLMRobertaTokenizer.from_pretrained("castorini/afriberta_base")
    af_model = AutoModelForSequenceClassification.from_pretrained(
        "castorini/afriberta_base",
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    )
    
    af_train_dataloader, af_eval_dataloader = tokenize_dataset(english_dataset, af_tokenizer)

    af_model = train_model(
        af_model, 
        af_train_dataloader, 
        af_eval_dataloader, 
        device, 
        "afriberta_english", 
        output_dir, 
        num_epochs=3
    )
    
    logger.info("Training completed. Models saved to saved-models directory.")

if __name__ == "__main__":
    main()
