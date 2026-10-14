import numpy as np

def compute_cosine_similarity(feature1, feature2):
    """
    Compute cosine similarity between two feature vectors or matrices.
    """
    if len(feature1.shape) == 1:
        feature1 = feature1.reshape(1, -1)
    if len(feature2.shape) == 1:
        feature2 = feature2.reshape(1, -1)
        
    dot_product = np.dot(feature1, feature2.T)
    norm1 = np.linalg.norm(feature1, axis=1).reshape(-1, 1)
    norm2 = np.linalg.norm(feature2, axis=1).reshape(1, -1)
    
    # Avoid division by zero
    norm_product = norm1 * norm2
    norm_product[norm_product == 0] = 1e-6
    
    return dot_product / norm_product

def match_embeddings(query_features, gallery_features_dict, threshold=0.6):
    """
    query_features: list of features for new tracks
    gallery_features_dict: dict of {global_id: [list of gallery features]}
    Returns: list of matches [(query_idx, global_id, similarity_score)]
    """
    if not query_features or not gallery_features_dict:
        return []
        
    global_ids = list(gallery_features_dict.keys())
    
    matches = []
    
    for q_idx, q_feat in enumerate(query_features):
        best_score = -1.0
        best_gid = None
        
        for gid in global_ids:
            g_feats = np.array(gallery_features_dict[gid])
            if len(g_feats) == 0:
                continue
                
            sim_matrix = compute_cosine_similarity(q_feat, g_feats)
            # Use the max similarity across the gallery
            max_sim = np.max(sim_matrix)
            
            if max_sim > best_score:
                best_score = max_sim
                best_gid = gid
                
        if best_score >= threshold:
            matches.append((q_idx, best_gid, best_score))
            
    return matches
