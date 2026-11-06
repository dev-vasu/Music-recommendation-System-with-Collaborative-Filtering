"""
Collaborative Filtering Engine using Matrix Factorization (TruncatedSVD) & Item-Item Similarity
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """
    Implements Latent Factor Matrix Factorization (SVD) and Item-Item
    Collaborative Filtering over user-song interaction matrices.
    """

    def __init__(self, n_factors: int = 30, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_item_matrix = None
        self.sparse_matrix = None
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_id_map = {}
        self.rev_user_map = {}
        self.track_id_map = {}
        self.rev_track_map = {}
        self.all_track_indices = None
        self.n_users = 0
        self.n_tracks = 0

    def fit(self, interactions_df: pd.DataFrame, n_total_tracks: int):
        """
        Fits TruncatedSVD on the user-item interaction matrix.
        """
        self.n_tracks = n_total_tracks
        unique_users = sorted(interactions_df['user_id'].unique())
        self.n_users = len(unique_users)

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.rev_user_map = {idx: uid for uid, idx in self.user_id_map.items()}

        row_indices = [self.user_id_map[u] for u in interactions_df['user_id']]
        col_indices = interactions_df['track_index'].values
        data = interactions_df['rating'].values

        self.sparse_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_users, self.n_tracks),
            dtype=np.float32
        )

        # Decompose using Truncated SVD
        n_components = min(self.n_factors, self.n_users - 1, self.n_tracks - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(self.sparse_matrix)  # (n_users, n_components)
        self.item_factors = self.svd.components_.T  # (n_tracks, n_components)

        return self

    def recommend_for_user(
        self,
        user_id: int,
        df: pd.DataFrame,
        n_recommendations: int = 10,
        exclude_interacted: bool = True
    ) -> pd.DataFrame:
        """
        Recommends top tracks for a registered user using latent dot product.
        """
        if user_id not in self.user_id_map:
            # Fall back to globally top-rated tracks
            top_pop = df.nlargest(n_recommendations, 'popularity_score')
            return top_pop[['Artist', 'Track', 'vibe_name', 'popularity_score']].reset_index(drop=True)

        u_idx = self.user_id_map[user_id]
        user_vector = self.user_factors[u_idx]  # shape: (n_components,)

        # Dot product with all items
        predicted_scores = np.dot(self.item_factors, user_vector)

        if exclude_interacted:
            interacted_items = self.sparse_matrix[u_idx].indices
            predicted_scores[interacted_items] = -np.inf

        top_indices = np.argsort(predicted_scores)[::-1][:n_recommendations]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = df.loc[idx]
            results.append({
                'rank': rank,
                'track_index': int(idx),
                'Artist': row['Artist'],
                'Track': row['Track'],
                'collab_score': round(float(predicted_scores[idx]), 4),
                'vibe_name': row.get('vibe_name', 'General'),
                'popularity_score': round(float(row.get('popularity_score', 0.0)), 4),
                'Url_spotify': row.get('Url_spotify', ''),
                'Url_youtube': row.get('Url_youtube', '')
            })

        return pd.DataFrame(results)

    def recommend_similar_tracks(
        self,
        track_idx: int,
        df: pd.DataFrame,
        n_recommendations: int = 10
    ) -> pd.DataFrame:
        """
        Item-item collaborative filtering based on latent cosine similarity.
        """
        if track_idx >= self.n_tracks:
            raise IndexError("Track index out of bounds.")

        item_vec = self.item_factors[track_idx].reshape(1, -1)
        sims = cosine_similarity(item_vec, self.item_factors).flatten()
        sims[track_idx] = -1.0  # exclude self

        top_indices = np.argsort(sims)[::-1][:n_recommendations]

        results = []
        for idx in top_indices:
            row = df.loc[idx]
            results.append({
                'track_index': int(idx),
                'Artist': row['Artist'],
                'Track': row['Track'],
                'collab_similarity': round(float(sims[idx]), 4),
                'vibe_name': row.get('vibe_name', 'General'),
                'popularity_score': round(float(row.get('popularity_score', 0.0)), 4)
            })

        return pd.DataFrame(results)
