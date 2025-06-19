from transformers import AutoTokenizer, AutoModelForSequenceClassification, XLMRobertaTokenizer
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import sys
import os


from data import load_and_preprocess_data

def load_models(emotion_labels):
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
    
    return {
        'mb_model': mb_model,
        'mb_tokenizer': mb_tokenizer,
        'af_model': af_model,
        'af_tokenizer': af_tokenizer
    }

def evaluate_model(model, tokenizer, dataset, batch_size=8):
    all_predictions = []
    all_labels = []

    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        texts = batch['text']
        
        labels = batch['labels']
        
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    accuracy = accuracy_score([l for labels in all_labels for l in labels], 
                             [p for preds in all_predictions for p in preds])
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    per_class_f1 = []
    for i in range(len(emotion_labels)):
        class_labels = [label[i] for label in all_labels]
        class_preds = [pred[i] for pred in all_predictions]
        class_f1 = f1_score(class_labels, class_preds, zero_division=0)
        per_class_f1.append(class_f1)
    
    emotion_f1_dict = {emotion: f1 for emotion, f1 in zip(emotion_labels, per_class_f1)}
    
    return {
        'accuracy': accuracy,
        'f1_score': f1_macro,
        'per_class_f1': emotion_f1_dict,
        'classification_report': classification_report(all_labels, all_predictions, zero_division=0)
    }

def main():
    """Main function to run the complete evaluation process"""

    data_dict = load_and_preprocess_data()
    zulu_dataset = data_dict['zulu_dataset']
    pidgin_dataset = data_dict['pidgin_dataset']
    emotion_labels = data_dict['emotion_labels']
    
 
    models = load_models(emotion_labels)
    mb_model = models['mb_model']
    mb_tokenizer = models['mb_tokenizer']
    af_model = models['af_model']
    af_tokenizer = models['af_tokenizer']   
    print("Zulu development set evaluation:")
    mb_dev_results = evaluate_model(mb_model, mb_tokenizer, zulu_dataset['dev'])
    af_dev_results = evaluate_model(af_model, af_tokenizer, zulu_dataset['dev'])

    print("\nZulu test set evaluation (final results):")
    mb_test_results = evaluate_model(mb_model, mb_tokenizer, zulu_dataset['test'])
    af_test_results = evaluate_model(af_model, af_tokenizer, zulu_dataset['test'])

   
    print("\nPer-class F1 scores (Zulu):")
    print("mBERT:")
    for emotion, score in mb_test_results['per_class_f1'].items():
        print(f"  {emotion}: {score:.4f}")

    print("\nAfriBERTa:")
    for emotion, score in af_test_results['per_class_f1'].items():
        print(f"  {emotion}: {score:.4f}")

    print("\nModel Comparison (Zulu):")
    print(f"mBERT dev F1 score: {mb_dev_results['f1_score']}")
    print(f"mBERT F1 test score: {mb_test_results['f1_score']}")
    print(f"AfriBERTa dev F1 score: {af_dev_results['f1_score']}")
    print(f"AfriBERTa test score: {af_test_results['f1_score']}")   
    print("\nPidgin development set evaluation:")
    mb_dev_results = evaluate_model(mb_model, mb_tokenizer, pidgin_dataset['dev'])
    af_dev_results = evaluate_model(af_model, af_tokenizer, pidgin_dataset['dev'])

    print("\nPidgin test set evaluation (final results):")
    mb_test_results = evaluate_model(mb_model, mb_tokenizer, pidgin_dataset['test'])
    af_test_results = evaluate_model(af_model, af_tokenizer, pidgin_dataset['test'])

  
    print("\nPer-class F1 scores (Pidgin):")
    print("mBERT:")
    for emotion, score in mb_test_results['per_class_f1'].items():
        print(f"  {emotion}: {score:.4f}")

    print("\nAfriBERTa:")
    for emotion, score in af_test_results['per_class_f1'].items():
        print(f"  {emotion}: {score:.4f}")

    print("\nModel Comparison (Pidgin):")
    print(f"mBERT dev F1 score: {mb_dev_results['f1_score']}")
    print(f"mBERT F1 test score: {mb_test_results['f1_score']}")
    print(f"AfriBERTa dev F1 score: {af_dev_results['f1_score']}")
    print(f"AfriBERTa test score: {af_test_results['f1_score']}")


if __name__ == "__main__":
    main()
