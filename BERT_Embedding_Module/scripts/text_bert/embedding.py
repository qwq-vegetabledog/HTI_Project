import glob
import os
from pathlib import Path
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import SubtitleWrapper

import torch
from transformers import BertTokenizer, BertModel

import numpy as np

def embedding(base_path):
    embedding_path = os.path.join(base_path, 'embedding')
    text_path = os.path.join(base_path, 'tsv')

    # Create embedding directory if it doesn't exist
    os.makedirs(embedding_path, exist_ok=True)

    tsv_files = sorted(glob.glob(text_path + "/*.tsv"))
    
    # Load model and tokenizer once outside the loop (more efficient)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
    model.to(device)
    model.eval()
    
    for tsv_file in tsv_files:
        name = os.path.split(tsv_file)[1][:-4]
        
        if os.path.isfile(tsv_file):
            # SubtitleWrapper retorna listas de (start, end, word) ou (word, start, end)
            # Assumindo padrão Genea: row[0]=start, row[1]=end, row[2]=word
            subtitle = SubtitleWrapper(tsv_file).get()
        else:
            continue

        # Extraímos palavras e tempos separadamente para manter indices alinhados
        words = np.array([str(line[2]) for line in subtitle])
        start_times = np.array([float(line[0]) for line in subtitle])
        end_times = np.array([float(line[1]) for line in subtitle])

        phrases = []
        intervals = [] # Lista para guardar [inicio, fim] de cada frase
        
        text = ""
        current_start = None # Para marcar o inicio da frase atual
        endPhrases = [".", "?", "!"]

        # Split text into phrases based on sentence endings
        for index in range(words.shape[0]):
            word = words[index]
            s_time = start_times[index]
            e_time = end_times[index]

            if word == " " or word == "":
                continue
            
            # Se é a primeira palavra da frase, guarda o tempo
            if current_start is None:
                current_start = s_time

            if word[-1] in endPhrases or index == (words.shape[0] - 1):
                text += word
                phrases.append(text)
                
                # Salva o intervalo da frase completa
                intervals.append([current_start, e_time])
                
                text = ""
                current_start = None # Reseta para a próxima
            else:
                text += word + " "
        
        # List to store embeddings from all phrases
        all_embeddings = []
        
        for index in range(len(phrases)):
            phrase = phrases[index]
            # Skip empty phrases
            if not phrase.strip():
                # Precisamos manter a sincronia com intervals, então salvamos um dummy se pular
                # Ou removemos o intervalo correspondente. Vamos remover o intervalo.
                if index < len(intervals):
                    del intervals[index]
                continue
                
            # Add BERT special tokens
            marked_text = "[CLS] " + phrase + " [SEP]"

            # Tokenize the phrase
            tokenized_text = tokenizer.tokenize(marked_text)
            
            # Convert tokens to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            
            # Truncar se passar de 512 (limite do BERT) para evitar erro
            if len(indexed_tokens) > 512:
                indexed_tokens = indexed_tokens[:512]
            
            # Create attention mask and token type ids
            attention_mask = [1] * len(indexed_tokens)
            token_type_ids = [0] * len(indexed_tokens)  # For single sentence

            # Convert to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            attention_mask_tensor = torch.tensor([attention_mask]).to(device)
            token_type_ids_tensor = torch.tensor([token_type_ids]).to(device)

            # Get embeddings (no gradient calculation for inference)
            with torch.no_grad():
                outputs = model(tokens_tensor, 
                            attention_mask=attention_mask_tensor,
                            token_type_ids=token_type_ids_tensor)
                
                # Now hidden_states will be available
                all_hidden_states = outputs.hidden_states
                
                # Take only the last 4 layers
                last_four = all_hidden_states[-4:]  # tuple of 4 layers

                # Concatenate 768*4 = 3072-dim
                token_embeddings = torch.cat(last_four, dim=2)

            # Remove batch dimension -> [Seq_Len, 3072]
            token_embeddings = torch.squeeze(token_embeddings, dim=0)

            all_embeddings.append(token_embeddings.cpu().numpy())
            
        # Save embeddings for this file
        if not all_embeddings:
            print(f"No text for {name}, creating zero embedding")
            all_embeddings.append(np.zeros((1, 3072), dtype=np.float32))
            intervals.append([0.0, 1.0]) # Intervalo dummy

        # Save embeddings AND INTERVALS
        output_file = os.path.join(embedding_path, f"{name}_embeddings.npz")
        
        # Cria o dicionário com as chaves phrase_0, phrase_1...
        embeddings_dict = {f"phrase_{i}": all_embeddings[i] for i in range(len(all_embeddings))}
        
        # ADICIONA OS INTERVALOS
        embeddings_dict['intervals'] = np.array(intervals)
        
        np.savez(output_file, **embeddings_dict)
        print(f"Embeddings saved to: {output_file}")
        print(f"Total number of phrases: {len(all_embeddings)}")

if __name__ == '__main__':
    folders = [
        "/home/HTI_project/dataset/genea/genea2023_dataset/trn/interloctr",
        "/home/HTI_project/dataset/genea/genea2023_dataset/trn/main-agent",
        "/home/HTI_project/dataset/genea/genea2023_dataset/tst/interloctr",
        "/home/HTI_project/dataset/genea/genea2023_dataset/tst/main-agent",
        "/home/HTI_project/dataset/genea/genea2023_dataset/val/interloctr",
        "/home/HTI_project/dataset/genea/genea2023_dataset/val/main-agent"
    ]
    
    for f in folders:
        embedding(f)