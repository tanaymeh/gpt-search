import os
import time
import torch
import openai
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

class TextFileEncoder:
    def __init__(self, file_path: str, model: 'SentenceTransformer'):
        assert file_path.endswith(".txt"), "Only .txt files are supposed in TextFileEncoder"
        self.raw_file_reader = open(file_path, 'r')
        self.model = model
        self.num_lines = sum(1 for l in open(file_path, 'r') if l.strip(' \n') != '')

    def file_reader(self, file_reader: 'TextIOWrapper'):
        """
        Little generator that skips blank lines
        """
        for line in file_reader:
            line = line.rstrip()
            if line: yield line

    def single_encode(self) -> list:
        """
        Encodes the text file line by line
        """
        start = time.time()
        encoded_text = []
        sentences = []
        for line in tqdm(self.file_reader(self.raw_file_reader), total=self.num_lines):
            encoded_text.append(self.model.encode(line, show_progress_bar=False))
            sentences.append(line)
        end = time.time() - start
        print(f"[INFO] Encoding TXT file took {end/60:.4f} seconds")
        return torch.tensor(np.array(encoded_text)), sentences


    def _batch_encode(self, batch_size: int) -> list:
        """
        Encodes the text file in batches
        """
        assert self.num_lines % batch_size == 0, f"Can't make {batch_size} batches from the text file."
        encoded_text = []
        ctr = 0
        hoard = []
        for line in tqdm(self.file_reader(self.raw_file_reader), total=self.num_lines):
            if ctr == batch_size:
                enc = self.model.encode(hoard)
                encoded_text.extend(enc)
                ctr = 0
                hoard = []
            else:
                ctr += 1
                hoard.append(line)
        return encoded_text

def file_extract(model, data) -> torch.Tensor:
    encoded_text = []
    for line in tqdm(data, total=len(data)):
        encoded_text.append(model.encode(line, show_progress_bar=False))
    return torch.tensor(np.array(encoded_text))

def extract(model, text) -> torch.Tensor:
    return torch.tensor(np.array(model.encode(text, show_progress_bar=False)))

def get_similar_sentences(lines, query, source, k=5):
    """
    Finds similarities and returns the 
    """
    cos_scores = util.pytorch_cos_sim(query, source)[0]
    top_results = torch.topk(cos_scores, k=k)
    return [lines[top_results[1][i]] for i in range(k)]

def openai_inference(data, k=5, temperature=0.1):
    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "assistant",
            "content": f"You are a smart search engine program that has to answer queries. The database of information is very large in size so you only have access to a few top results that match a given query the most semantically. Given the top {k} results (in the order of their similarity): [{data['top_results']}], answer the given queries: {data['query']}\nIf you can't answer the query, only and only give the following reply: 'Unable to answer the query.'",
        }],
        temperature=temperature,
    )
    return out['choices'][0]['message']['content']