"""
Evaluation Metrics & Benchmarking Suite for Recommender Systems
Computes Precision@K, Recall@K, MAP@K, NDCG@K, Diversity, and Novelty.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_distances


def precision_at_k(recommended: List[int], ground_truth: List[int], k: int = 10) -> float:
    """Computes Precision at K."""
    if not recommended or not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return hits / k


def recall_at_k(recommended: List[int], ground_truth: List[int], k: int = 10) -> float:
    """Computes Recall at K."""
    if not recommended or not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return hits / len(ground_truth)


def average_precision_at_k(recommended: List[int], ground_truth: List[int], k: int = 10) -> float:
    """Computes Average Precision (AP) at K for a single query."""
    if not recommended or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    hits = 0
    score = 0.0

    for i, item in enumerate(recommended[:k]):
        if item in gt_set:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(gt_set), k) if gt_set else 0.0


def mean_average_precision_at_k(
    predictions: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int = 10
) -> float:
    """Computes Mean Average Precision (MAP) at K across users."""
    ap_scores = [
        average_precision_at_k(predictions[u], ground_truth[u], k)
        for u in ground_truth if u in predictions and len(ground_truth[u]) > 0
    ]
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def ndcg_at_k(recommended: List[int], ground_truth: List[int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG) at K."""
    if not recommended or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return (dcg / idcg) if idcg > 0 else 0.0


def intra_list_diversity(recommended_indices: List[int], feature_matrix: np.ndarray) -> float:
    """
    Computes Intra-List Diversity (ILD) as the average pairwise cosine distance
    between recommended tracks. High ILD = varied soundscape; low ILD = uniform sound.
    """
    if len(recommended_indices) < 2:
        return 0.0

    valid_idx = [i for i in recommended_indices if i < len(feature_matrix)]
    if len(valid_idx) < 2:
        return 0.0

    vecs = feature_matrix[valid_idx]
    dist_matrix = cosine_distances(vecs)
    # Average of upper triangular elements (excluding diagonal)
    n = len(valid_idx)
    upper_sum = np.sum(np.triu(dist_matrix, k=1))
    n_pairs = n * (n - 1) / 2
    return float(upper_sum / n_pairs) if n_pairs > 0 else 0.0


def novelty_score(recommended_indices: List[int], popularity_scores: np.ndarray) -> float:
    """
    Measures playlist novelty / serendipity.
    Novelty = 1 - mean(popularity of recommended tracks).
    Higher novelty means exploring deeper catalog beyond mainstream hits.
    """
    if not recommended_indices:
        return 0.0
    pops = [popularity_scores[i] for i in recommended_indices if i < len(popularity_scores)]
    return float(1.0 - np.mean(pops)) if pops else 0.0


def run_benchmark(
    df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    k: int = 10
) -> pd.DataFrame:
    """
    Performs train/test split on user interactions, generates recommendations
    from Content-Based, Collaborative, and Hybrid engines, and returns a
    comparative evaluation scorecard.
    """
    from .recommenders.content_based import ContentBasedRecommender
    from .recommenders.collaborative import CollaborativeRecommender
    from .recommenders.hybrid import HybridRecommender

    # Train / test split by holding out 20% of interactions per user
    train_list = []
    test_dict = {}

    for uid, group in interactions_df.groupby('user_id'):
        tracks = group['track_index'].tolist()
        if len(tracks) >= 5:
            split_idx = int(len(tracks) * 0.8)
            train_list.append(group.iloc[:split_idx])
            test_dict[uid] = tracks[split_idx:]
        else:
            train_list.append(group)

    train_df = pd.concat(train_list).reset_index(drop=True)

    # Train models
    cb = ContentBasedRecommender().fit(df)
    cf = CollaborativeRecommender(n_factors=25).fit(train_df, len(df))
    hybrid = HybridRecommender(cb, cf, df)

    eval_users = list(test_dict.keys())[:150]  # evaluate on 150 held-out users
    cb_preds, cf_preds, hy_preds = {}, {}, {}

    for u in eval_users:
        user_history = train_df[train_df['user_id'] == u]['track_index'].tolist()
        if not user_history:
            continue
        seed_track = user_history[-1]

        # Content-Based
        cb_res = cb.recommend_for_track(seed_track, n_recommendations=k)
        cb_preds[u] = cb_res['track_index'].tolist() if not cb_res.empty else []

        # Collaborative
        cf_res = cf.recommend_for_user(u, df, n_recommendations=k, exclude_interacted=True)
        cf_preds[u] = cf_res['track_index'].tolist() if not cf_res.empty else []

        # Hybrid
        hy_res = hybrid.recommend(seed_track, user_id=u, alpha=0.55, beta=0.30, gamma=0.15, n_recommendations=k)
        hy_preds[u] = hy_res['track_index'].tolist() if not hy_res.empty else []

    feature_matrix = cb.feature_matrix
    pop_scores = df['popularity_score'].to_numpy()

    # Calculate metrics for each model
    results = []
    for model_name, preds in [
        ("Content-Based (Audio Features)", cb_preds),
        ("Collaborative (Truncated SVD)", cf_preds),
        ("Hybrid (Acoustic + Latent + Popularity)", hy_preds)
    ]:
        precisions = [precision_at_k(preds.get(u, []), test_dict[u], k) for u in eval_users if u in preds]
        recalls = [recall_at_k(preds.get(u, []), test_dict[u], k) for u in eval_users if u in preds]
        map_val = mean_average_precision_at_k(preds, test_dict, k)
        ndcgs = [ndcg_at_k(preds.get(u, []), test_dict[u], k) for u in eval_users if u in preds]
        divs = [intra_list_diversity(preds.get(u, []), feature_matrix) for u in eval_users if u in preds]
        novs = [novelty_score(preds.get(u, []), pop_scores) for u in eval_users if u in preds]

        results.append({
            "Model": model_name,
            f"Precision@{k}": round(float(np.mean(precisions)), 4),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
            f"MAP@{k}": round(map_val, 4),
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
            "Intra-List Diversity": round(float(np.mean(divs)), 4),
            "Catalog Novelty": round(float(np.mean(novs)), 4)
        })

    return pd.DataFrame(results)
