from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import XLMRobertaTokenizer
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
login(token=token)

brighter_eng = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng")
brighter_zul = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "zul")
brighter_pcm = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "pcm")

afrisenti_pcm = load_dataset("shmuhammad/AfriSenti-twitter-sentiment", "pcm")

afrihate_pcm = load_dataset("afrihate/afrihate", "pcm")
afrihate_zul = load_dataset("afrihate/afrihate", "zul")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

brighter_zul = brighter_zul.map(lambda x: {"text": clean_text(x["text"])})
brighter_pcm = brighter_pcm.map(lambda x: {"text": clean_text(x["text"])})

# Define emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def preprocess(example):
    labels = [example[e] for e in emotion_labels]
    return {'labels': labels}

zulu_dataset = brighter_zul.map(preprocess)
pidgin_dataset = brighter_pcm.map(preprocess)

train = brighter_pcm['train']
val = brighter_pcm['dev']
test = brighter_pcm['test']

# print(train[:5])

# mBERT
mb_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
mb_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(emotion_labels),
    problem_type="multi_label_classification"
)


# AfriBERTa
af_tokenizer = XLMRobertaTokenizer.from_pretrained("castorini/afriberta_base")
af_model = AutoModelForSequenceClassification.from_pretrained(
    "castorini/afriberta_base",
    num_labels=len(emotion_labels),
    problem_type="multi_label_classification"
)

mb_model.eval()
af_model.eval()


def evaluate_model(model, tokenizer, dataset, batch_size=8):
    all_predictions = []
    all_labels = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        texts = batch['text']
        
        # Access labels directly from batch
        labels = batch['labels']
        
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    # Calculate overall metrics 
    accuracy = accuracy_score([l for labels in all_labels for l in labels], 
                             [p for preds in all_predictions for p in preds])
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # Calculate per-class F1 scores
    per_class_f1 = []
    for i in range(len(emotion_labels)):
        class_labels = [label[i] for label in all_labels]
        class_preds = [pred[i] for pred in all_predictions]
        class_f1 = f1_score(class_labels, class_preds, zero_division=0)
        per_class_f1.append(class_f1)
    
    # Create dictionary of per-class F1 scores
    emotion_f1_dict = {emotion: f1 for emotion, f1 in zip(emotion_labels, per_class_f1)}
    
    return {
        'accuracy': accuracy,
        'f1_score': f1_macro,
        'per_class_f1': emotion_f1_dict,
        'classification_report': classification_report(all_labels, all_predictions, zero_division=0)
    }

#Zulu

# Development evaluation
print("Zulu development set evaluation:")
mb_dev_results = evaluate_model(mb_model, mb_tokenizer, zulu_dataset['dev'])
af_dev_results = evaluate_model(af_model, af_tokenizer, zulu_dataset['dev'])

# After finalizing your approach:
print("\nZulu test set evaluation (final results):")
mb_test_results = evaluate_model(mb_model, mb_tokenizer, zulu_dataset['test'])
af_test_results = evaluate_model(af_model, af_tokenizer, zulu_dataset['test'])

# Compare results
print("\nPer-class F1 scores (Zulu):")
print("mBERT:")
for emotion, score in mb_test_results['per_class_f1'].items():
    print(f"  {emotion}: {score:.4f}")

print("\nAfriBERTa:")
for emotion, score in af_test_results['per_class_f1'].items():
    print(f"  {emotion}: {score:.4f}")

print("\nModel Comparison:")
print(f"mBERT dev F1 score: {mb_dev_results['f1_score']}")
print(f"mBERT F1 test score: {mb_test_results['f1_score']}")
print(f"AfriBERTa dev F1 score: {af_dev_results['f1_score']}")
print(f"AfriBERTa test score: {af_test_results['f1_score']}")


#Pidgin

# Development evaluation
print("Pidgin development set evaluation:")
mb_dev_results = evaluate_model(mb_model, mb_tokenizer, pidgin_dataset['dev'])
af_dev_results = evaluate_model(af_model, af_tokenizer, pidgin_dataset['dev'])

# After finalizing your approach:
print("\nPidgin test set evaluation (final results):")
mb_test_results = evaluate_model(mb_model, mb_tokenizer, pidgin_dataset['test'])
af_test_results = evaluate_model(af_model, af_tokenizer, pidgin_dataset['test'])

# Compare results
print("\nPer-class F1 scores (Zulu):")
print("mBERT:")
for emotion, score in mb_test_results['per_class_f1'].items():
    print(f"  {emotion}: {score:.4f}")

print("\nAfriBERTa:")
for emotion, score in af_test_results['per_class_f1'].items():
    print(f"  {emotion}: {score:.4f}")

print("\nModel Comparison:")
print(f"mBERT dev F1 score: {mb_dev_results['f1_score']}")
print(f"mBERT F1 test score: {mb_test_results['f1_score']}")
print(f"AfriBERTa dev F1 score: {af_dev_results['f1_score']}")
print(f"AfriBERTa test score: {af_test_results['f1_score']}")