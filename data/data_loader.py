from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
import re
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_text(text):
    """Clean text by lowercasing and removing URLs and punctuation"""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def load_and_preprocess_data():
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

    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    brighter_eng = brighter_eng.map(lambda x: {"text": clean_text(x["text"])})
    brighter_zul = brighter_zul.map(lambda x: {"text": clean_text(x["text"])})
    brighter_pcm = brighter_pcm.map(lambda x: {"text": clean_text(x["text"])})

    def preprocess(example):
        labels = [example[e] for e in emotion_labels]
        return {'labels': labels}

    english_dataset = brighter_eng.map(preprocess)
    zulu_dataset = brighter_zul.map(preprocess)
    pidgin_dataset = brighter_pcm.map(preprocess)

    return {
        'english_dataset': english_dataset,
        'zulu_dataset': zulu_dataset,
        'pidgin_dataset': pidgin_dataset,
        'emotion_labels': emotion_labels
    }

if __name__ == "__main__":
    data = load_and_preprocess_data()
    print(f"English dataset size: {len(data['english_dataset']['train'])}")
    print(f"Zulu dataset size: {len(data['zulu_dataset']['train'])}")
    print(f"Pidgin dataset size: {len(data['pidgin_dataset']['train'])}")
    print(f"Emotion labels: {data['emotion_labels']}")
