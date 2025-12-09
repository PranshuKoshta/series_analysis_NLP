# try:
#     import spacy
# except:
#     spacy = None
# from nltk.tokenize import sent_tokenize
# import pandas as pd
# from ast import literal_eval
# import os 
# import sys
# import pathlib
# folder_path = pathlib.Path().parent.resolve()
# sys.path.append(os.path.join(folder_path, '../'))
# from utils import load_subtitles_dataset

# class NamedEntityRecognizer:
#     def __init__(self):
#         self.nlp_model = self.load_model()
#         pass

#     def load_model(self):
#         nlp = spacy.load("en_core_web_trf")
#         return nlp

#     def get_ners_inference(self,script):
#         script_sentences = sent_tokenize(script)

#         ner_output = []

#         for sentence in script_sentences:
#             doc = self.nlp_model(sentence)
#             ners = set()
#             for entity in doc.ents:
#                 if entity.label_ =="PERSON":
#                     full_name = entity.text
#                     first_name = full_name.split(" ")[0]
#                     first_name = first_name.strip()
#                     ners.add(first_name)
#             ner_output.append(ners)

#         return ner_output

#     def get_ners(self,dataset_path,save_path=None):
#         if save_path is not None and os.path.exists(save_path):
#             df = pd.read_csv(save_path)
#             df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
#             return df

#         # load dataset 
#         df = load_subtitles_dataset(dataset_path)

#         # Run Inference
#         df['ners'] = df['script'].apply(self.get_ners_inference)

#         if save_path is not None:
#             df.to_csv(save_path,index=False)
        
#         return df

try:
    import spacy
except:
    spacy = None
from nltk.tokenize import sent_tokenize
import pandas as pd
from ast import literal_eval
import os 
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = None
        

    def load_model(self):
        if spacy is None:
            return None
        try:
            nlp = spacy.load("en_core_web_trf")
            return nlp
        except Exception:
            return None

    def get_ners_inference(self, script):
        # Lazy-load the spaCy model if needed
        if self.nlp_model is None:
           self.nlp_model = self.load_model()

        if self.nlp_model is None:
            raise RuntimeError(
                    "spaCy transformer model not available. "
                    "On Hugging Face Spaces provide a precomputed NER CSV and pass its path as save_path."
           )

        script_sentences = sent_tokenize(script)
        ner_output = []
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    first_name = entity.text.split(" ")[0].strip()
                    ners.add(first_name)
            ner_output.append(ners)
        return ner_output

    def get_ners(self,dataset_path,save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
            return df

        # load dataset 
        df = load_subtitles_dataset(dataset_path)

        # Run Inference
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path,index=False)
        
        return df