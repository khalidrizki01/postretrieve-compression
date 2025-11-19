import gc
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..') 
from all_utils import average_pool
from datasets import DatasetDict
import random

def add_negative_passages(batch, indices, embedding_tokenizer, embedding_model, index, corpus_docids, corpus_dict):
#     batch_queries = batch["query"]  # List of queries
    batch_queries = [f"query: {query}" for query in batch["query"]]

    batch_dict = embedding_tokenizer(batch_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.to("cuda:0") for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = embedding_model(**batch_dict)

    query_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1).cpu().numpy().astype(np.float32)  # (batch, dim)

    # FAISS Search untuk semua query dalam batch
    D, I = index.search(query_embeddings, 7)  # Ambil 7 kandidat

    # Iterasi untuk setiap query dalam batch
    negative_passages_batch = []
    for i, idx in enumerate(indices):
        positive_docids = set(p["docid"] for p in batch["positive_passages"][i])
        num_positive = len(positive_docids)

        # Tentukan jumlah negatif sesuai aturan:
        if num_positive == 1:
            max_negatives = 2
        elif num_positive == 2:
            max_negatives = 1
        else:  # 3 atau lebih
            max_negatives = 0
        selected_negative_passages = []

        for doc_idx in I[i]:  # Loop hasil FAISS untuk query ke-i
            if max_negatives == 0:
                break  # Tidak perlu ambil negatif
            docid = corpus_docids[doc_idx]
            if docid not in positive_docids:
                title, text = corpus_dict[docid]
                selected_negative_passages.append({"docid": docid, "title": title, "text": text})
            if len(selected_negative_passages) == max_negatives:
                break

        negative_passages_batch.append(selected_negative_passages)

    batch["negative_passages"] = negative_passages_batch

    # 🔥 **BERSIHKAN CACHE GPU & MEMORI SETELAH BATCH SELESAI**
    del batch_dict, outputs, query_embeddings
    torch.cuda.empty_cache()  # Kosongkan cache GPU
    gc.collect()  # Kosongkan cache CPU untuk menghindari memory leak

    return batch

def select_top2_negative_passages(example, embedding_tokenizer, embedding_model):
    query_text = f'query: {example["query"]}'
    negative_passages = example["negative_passages"]

    # Jika sudah <= 2, tidak perlu pemrosesan
    if len(negative_passages) <= 2:
        return example

    # Ambil teks dari negative_passages
    neg_texts = [f'passage: {neg["title"]} | {neg['text']}' for neg in negative_passages]

    # Tokenisasi dan embedding query serta negative_passages
    batch_dict = embedding_tokenizer([query_text] + neg_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.to("cuda:0") for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = embedding_model(**batch_dict)

    # Hitung embedding dan normalisasi
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalisasi untuk cosine similarity

    # Hitung similarity scores (query vs negative_passages)
    query_embedding = embeddings[0].unsqueeze(0)  # Query ada di indeks pertama
    neg_embeddings = embeddings[1:]  # Negative passages setelah query
    scores = (query_embedding @ neg_embeddings.T).squeeze(0)  # Cosine similarity

    # Ambil indeks top 2 dengan similarity tertinggi
    top_indices = torch.argsort(scores, descending=True)[:2]

    # Simpan hanya 2 negative_passages terbaik
    example["negative_passages"] = [negative_passages[i] for i in top_indices]

    # Bersihkan cache GPU setelah query diproses
    del batch_dict, outputs, embeddings, scores
    torch.cuda.empty_cache()
    gc.collect()

    return example

def create_top_3_passages(example):
    # Mengambil positive_passages dan negative_passages
    positive_passages = example["positive_passages"]
    negative_passages = example["negative_passages"]

    # Gabungkan 3 passages sesuai dengan aturan yang diinginkan
    if len(positive_passages) == 3:
        top_3_passages = positive_passages
    elif len(positive_passages) == 2:
        top_3_passages = positive_passages + [negative_passages[0]]  # Ambil negative pertama
    elif len(positive_passages) == 1:
        top_3_passages = positive_passages + negative_passages[:2]  # Ambil 2 negative pertama
    else:
        top_3_passages = []  # Default jika tidak sesuai dengan aturan

    example["top_3_passages"] = top_3_passages
    return example