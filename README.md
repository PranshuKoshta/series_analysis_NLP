
---
title: TV Series Analyser
emoji: 📺
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# TV Series Analyser



# AI-Powered TV Series Analyzer

This project is an interactive web application that uses multiple NLP models to perform an in-depth analysis of a TV series. The system can scrape custom data, analyze character relationships, classify text, and identify narrative themes, all presented through a user-friendly Gradio interface.

## Features

**Custom Text Classification:** Fine-tunes a DistilBERT model on data acquired via a Scrapy web crawler to classify domain-specific text (e.g., classifying ninja attack types in Naruto).

**Character Network Analysis:** Uses SpaCy for Named Entity Recognition (NER) and NetworkX/PyViz to build and visualize an interactive graph of character relationships and interaction strength.

**Dynamic Theme Analysis:** Applies a Zero-Shot classification model to identify and score the prevalence of various user-defined themes throughout the series dialogue.


## Tech Stack

**Python**

**PyTorch**

**NLP Libraries:** Hugging Face Transformers, SpaCy

**Web Scraping:** Scrapy

PranshuKoshta/series_analysis_NLP NetworkX, PyViz

PranshuKoshta/series_analysis_NLP Gradio

**Data Handling:** Pandas

## How to Run

1. Clone the repository:
```
git clone https://github.com/PranshuKoshta/series_analysis_NLP.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Launch the Gradio application:
```
python app.py

```
