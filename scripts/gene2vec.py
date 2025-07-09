def load_gene_embeddings(file_path):
    gene_dict = {}
    with open(file_path, 'r') as file:
        next(file)  
        for line in file:
            elements = line.strip().split() 
            gene_name = elements[0]
            embedding = [float(x) for x in elements[1:]] 
            gene_dict[gene_name] = embedding
    return gene_dict

if __name__ == '__main__':
    file_path = r'sh\scripts\MultiDimGCN\biological database\gene2vec_dim_200_iter_9_w2v.txt'
    gene_embeddings = load_gene_embeddings(file_path)
    gene_name = 'PLAC4'
    embedding = gene_embeddings.get(gene_name)
    if embedding:
        print(f"The embedding for gene {gene_name} is {embedding}")
    else:
        print(f"Gene {gene_name} not found in the file.")
