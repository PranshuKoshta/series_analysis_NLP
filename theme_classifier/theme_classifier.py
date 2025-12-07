
from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import numpy as np
import os
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset
nltk.download('punkt', quiet=True) # Added quiet=True to reduce terminal spam

class ThemeClassifier():
    # CHANGED: __init__ no longer takes theme_list
    def __init__(self):
        print("Initializing ThemeClassifier and loading model...")
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = self.load_model(self.device)
        print("Model loaded successfully.")
    
    def load_model(self, device): 
        return pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )

    # CHANGED: This function now accepts theme_list as an argument
    def get_themes_inference(self, script, theme_list):
        script_sentences = sent_tokenize(script)
        
        # Batch Sentence
        sentence_batch_size = 20
        script_batches = [
            " ".join(script_sentences[i:i + sentence_batch_size])
            for i in range(0, len(script_sentences), sentence_batch_size)
        ]
        
        # Run Model 
        theme_output = self.classifier(
            script_batches,
            theme_list,
            multi_label=True
        )


        # wrangle outputs
        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        return {key: np.mean(np.array(value)) for key, value in themes.items()}

    # CHANGED: This function now accepts theme_list and passes it down
    def get_themes(self, dataset_path, theme_list, save_path=None):
        if save_path and os.path.exists(save_path):
            return pd.read_csv(save_path)

        df = load_subtitles_dataset(dataset_path)

        df = df.iloc[:15].copy() # For testing purposes, limit to first row

        # Use a lambda function to pass the extra theme_list argument
        output_themes = df['script'].apply(lambda s: self.get_themes_inference(s, theme_list))

        themes_df = pd.DataFrame(output_themes.tolist())
        df = pd.concat([df, themes_df], axis=1)

        if save_path:
            df.to_csv(save_path, index=False)
        
        return df