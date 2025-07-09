import re
import time
import mygene
import pickle
import requests
import scanpy as sc
import multiprocessing as mp

from goatools.obo_parser import GODag
from tqdm import tqdm
from retrying import retry
from collections import defaultdict
from itertools import combinations
from typing import Set
import pandas as pd
import os

mg = mygene.MyGeneInfo()
global_global_ensembl_ids_mapping = {}
global_global_go_terms_mapping = {}
global_global_go_dag = None

def init_worker(ensembl_ids_mapping, go_terms_mapping, go_dag):
    
    global global_global_ensembl_ids_mapping
    global global_global_go_terms_mapping
    global global_global_go_dag
    global_global_ensembl_ids_mapping = ensembl_ids_mapping
    global_global_go_terms_mapping = go_terms_mapping
    global_global_go_dag = go_dag

def retry_if_ssl_or_connection_error_or_429(exception):
    if isinstance(exception, requests.exceptions.HTTPError) and exception.response.status_code == 429:
        return True
    return isinstance(exception, (requests.exceptions.SSLError, requests.exceptions.ConnectionError))

@retry(retry_on_exception=retry_if_ssl_or_connection_error_or_429, stop_max_attempt_number=5, wait_fixed=2000) # 每次重试等待2000毫秒（2秒）
def fetch_go_terms(ensembl_id):
    try:
        result = mg.getgene(ensembl_id, fields='go')
        go_terms = []

        if 'go' in result:
            for category in ['BP', 'MF', 'CC']:
                terms = result['go'].get(category, [])
                if isinstance(terms, dict):
                    go_terms.append(terms['id'])
                elif isinstance(terms, list):
                    go_terms.extend(term['id'] for term in terms)
            
        time.sleep(10) 
        return ensembl_id, go_terms
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            time.sleep(1) 
            raise
        else:
            raise

def get_ensembl_ids(genes, species):
    ensembl_mapping = defaultdict(list)
    result = mg.querymany(genes, scopes='symbol', fields='ensembl.gene', species=species)
    missing_gene = []
    for item in result:
        ensembl_data = item.get('ensembl', None)
        if ensembl_data:
            if isinstance(ensembl_data, list):
                ensembl_ids = [ens['gene'] for ens in ensembl_data if 'gene' in ens]
                ensembl_mapping[item['query']].extend(ensembl_ids)
            elif isinstance(ensembl_data, dict):
                gene_id = ensembl_data.get('gene')
                if gene_id:
                    ensembl_mapping[item['query']].append(gene_id)
                else:
                    ensembl_mapping[item['query']].append('Not Found')
                    missing_gene.append(item['query'])
        else:
            ensembl_mapping[item['query']].append('Not Found')
            missing_gene.append(item['query'])
    if missing_gene:
        print(f"Ensembl IDs not found for the following genes: {missing_gene}")
        
    return ensembl_mapping, missing_gene

def prefetch_go_terms(ensembl_ids, num_workers=8):
    go_terms_mapping = {}
    
    with mp.Pool(num_workers) as pool:
        for ensembl_id, go_terms in tqdm(pool.imap_unordered(fetch_go_terms, ensembl_ids), total=len(ensembl_ids), desc="Fetching GO terms"):
            go_terms_mapping[ensembl_id] = go_terms
            
    return go_terms_mapping

def calculate_gene_similarity(similarity_scores, go_terms1, go_terms2):
    n_i = len(go_terms1)
    n_j = len(go_terms2)

    if n_i == 0 or n_j == 0:
        return 0.0

    n_ij = sum(1 for score in similarity_scores.values() if score > 0)
    if n_ij == 0:
        return 0.0

    similarity_score = sum(similarity_scores.values()) / n_ij
    return similarity_score

def calculate_go_term_similarity(go_terms1, go_terms2, go_dag):
    similarities = defaultdict(float)
    for go1 in go_terms1:
        for go2 in go_terms2:
            if go1 in go_dag and go2 in go_dag:
                if go_dag[go1].namespace == go_dag[go2].namespace:
                    intersect = len(set(go_dag[go1].get_all_parents()) & set(go_dag[go2].get_all_parents()))
                    length1 = go_dag[go1].depth
                    length2 = go_dag[go2].depth
                    if length1 > 0 and length2 > 0:
                        tmk = intersect / max(length1, length2)
                    else:
                        tmk = 0.0
                    similarities[(go1, go2)] = tmk
                else:
                    similarities[(go1, go2)] = 0.0
                    
    return similarities

def compute_gene_pair_similarity(gene_pair):
    gene1, gene2 = gene_pair
    best_similarity = 0.0

    for ensembl1 in global_global_ensembl_ids_mapping.get(gene1, []):
        if ensembl1 == 'Not Found':
            continue
        go_ids1 = global_global_go_terms_mapping.get(ensembl1, [])
        for ensembl2 in global_global_ensembl_ids_mapping.get(gene2, []):
            if ensembl2 == 'Not Found':
                continue
            go_ids2 = global_global_go_terms_mapping.get(ensembl2, [])

            similarity_scores = calculate_go_term_similarity(go_ids1, go_ids2, global_global_go_dag)
            similarity = calculate_gene_similarity(similarity_scores, go_ids1, go_ids2)

            if similarity > best_similarity:
                best_similarity = similarity

    return (gene1, gene2, best_similarity)



def compute_all_gene_similarities_parallel(gene_pairs, ensembl_ids_mapping, go_terms_mapping, go_dag, output_dir, num_workers=4):
    print(f"Number of CPU cores currently in use: {num_workers}")
    with mp.Pool(num_workers, initializer=init_worker, initargs=(ensembl_ids_mapping, go_terms_mapping, go_dag)) as pool:
        results = list(
            tqdm(pool.imap(compute_gene_pair_similarity, gene_pairs), total=len(gene_pairs), desc="Computing similarities")
        )
    gene_similarity_matrix = defaultdict(dict)
    for gene1, gene2, similarity in results:
        gene_similarity_matrix[gene1][gene2] = similarity
        gene_similarity_matrix[gene2][gene1] = similarity  

    return gene_similarity_matrix

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

