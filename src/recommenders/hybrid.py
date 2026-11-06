"""
Hybrid Music Recommendation Engine
Blends Content-Based Acoustic Matching, Collaborative Latent Factors, and Popularity Signals
"""

import numpy as np
import pandas as pd
from typing import Optional
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Blends acoustic similarity, collaborative filtering latent interactions,
    and popularity bias with dynamically tunable weights.
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collab_model: CollaborativeRecommender,
        df: pd.DataFrame
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.df = df.reset_index(drop=True)

    def recommend(
        self,
        seed_track_idx: int,
        user_id: Optional[int] = None,
        alpha: float = 0.55,  # Acoustic similarity weight
        beta: float = 0.30,   # Collaborative taste weight
        gamma: float = 0.15,  # Popularity weight
        n_recommendations: int = 10,
        vibe_filter: Optional[str] = None,
        filter_same_artist: bool = False
    ) -> pd.DataFrame:
        """
        Generates hybrid recommendations balancing acoustic coherence,
        crowd listening patterns, and popularity.
        """
        n_tracks = len(self.df)
        seed_row = self.df.loc[seed_track_idx]
        seed_artist = seed_row['Artist']

        # 1. Content-based acoustic similarities
        seed_vec = self.content_model.feature_matrix[seed_track_idx].reshape(1, -1)
        acoustic_sims = 1.0 - self.content_model.nn_model.metric_func(
            seed_vec, self.content_model.feature_matrix
        ) if hasattr(self.content_model.nn_model, 'metric_func') else None

        if acoustic_sims is None:
            from sklearn.metrics.pairwise import cosine_similarity
            acoustic_sims = cosine_similarity(seed_vec, self.content_model.feature_matrix).flatten()

        # 2. Collaborative similarities
        if self.collab_model.item_factors is not None and seed_track_idx < len(self.collab_model.item_factors):
            from sklearn.metrics.pairwise import cosine_similarity
            item_vec = self.collab_model.item_factors[seed_track_idx].reshape(1, -1)
            collab_sims = cosine_similarity(item_vec, self.collab_model.item_factors).flatten()
            # Normalize to [0, 1]
            collab_sims = (collab_sims + 1.0) / 2.0
        else:
            collab_sims = np.zeros(n_tracks, dtype=np.float32)

        # 3. Popularity signal
        pop_scores = self.df['popularity_score'].to_numpy(dtype=np.float32)

        # Normalize weights so they sum to 1.0
        total_weight = alpha + beta + gamma
        if total_weight > 0:
            a, b, g = alpha / total_weight, beta / total_weight, gamma / total_weight
        else:
            a, b, g = 0.6, 0.3, 0.1

        hybrid_scores = a * acoustic_sims + b * collab_sims + g * pop_scores
        hybrid_scores[seed_track_idx] = -1.0  # exclude the seed track itself

        # Candidate ranking
        ranked_indices = np.argsort(hybrid_scores)[::-1]

        results = []
        for idx in ranked_indices:
            row = self.df.loc[idx]

            if filter_same_artist and row['Artist'] == seed_artist:
                continue

            if vibe_filter and vibe_filter != "All" and row.get('vibe_name') != vibe_filter:
                continue

            results.append({
                'rank': len(results) + 1,
                'track_index': int(idx),
                'Artist': row['Artist'],
                'Track': row['Track'],
                'Album': row.get('Album', ''),
                'hybrid_score': round(float(hybrid_scores[idx]), 4),
                'acoustic_match': round(float(acoustic_sims[idx]), 4),
                'collab_match': round(float(collab_sims[idx]), 4),
                'popularity': round(float(pop_scores[idx]), 4),
                'vibe_name': row.get('vibe_name', 'General'),
                'Url_spotify': row.get('Url_spotify', ''),
                'Url_youtube': row.get('Url_youtube', '')
            })

            if len(results) >= n_recommendations:
                break

        return pd.DataFrame(results)
