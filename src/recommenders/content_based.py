"""
Content-Based Recommendation Engine using Spotify Acoustic Audio Features
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from ..preprocessing import NORM_AUDIO_FEATURES


class ContentBasedRecommender:
    """
    Recommends songs based on acoustic similarity (tempo, valence, energy,
    danceability, acousticness, instrumentalness, speechiness, etc.).
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.feature_matrix = None
        self.df = None
        self.nn_model = None

    def fit(self, df: pd.DataFrame):
        """Fits the nearest neighbors model on normalized acoustic features."""
        self.df = df.reset_index(drop=True)
        self.feature_matrix = self.df[NORM_AUDIO_FEATURES].to_numpy(dtype=np.float32)

        self.nn_model = NearestNeighbors(
            n_neighbors=min(50, len(self.df)),
            metric=self.metric,
            algorithm="brute"
        )
        self.nn_model.fit(self.feature_matrix)
        return self

    def recommend_for_track(
        self,
        track_idx: int,
        n_recommendations: int = 10,
        filter_same_artist: bool = False,
        vibe_filter: str = None
    ) -> pd.DataFrame:
        """
        Recommends top-N most similar songs for a given track index.
        """
        if self.nn_model is None:
            raise ValueError("Model is not fitted. Call fit(df) first.")

        query_vec = self.feature_matrix[track_idx].reshape(1, -1)
        k = min(n_recommendations * 4 + 1, len(self.df))
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=k)

        cand_indices = indices.flatten()[1:]
        cand_distances = distances.flatten()[1:]

        seed_artist = self.df.loc[track_idx, 'Artist']

        results = []
        for idx, dist in zip(cand_indices, cand_distances):
            row = self.df.loc[idx]
            if filter_same_artist and row['Artist'] == seed_artist:
                continue
            if vibe_filter and row.get('vibe_name') != vibe_filter:
                continue

            sim_score = max(0.0, 1.0 - float(dist))
            results.append({
                'track_index': int(idx),
                'Artist': row['Artist'],
                'Track': row['Track'],
                'Album': row.get('Album', ''),
                'similarity_score': round(sim_score, 4),
                'vibe_name': row.get('vibe_name', 'General'),
                'popularity_score': round(float(row.get('popularity_score', 0.0)), 4),
                'Url_spotify': row.get('Url_spotify', ''),
                'Url_youtube': row.get('Url_youtube', '')
            })

            if len(results) >= n_recommendations:
                break

        return pd.DataFrame(results)

    def recommend_for_profile(
        self,
        target_features: dict,
        n_recommendations: int = 10
    ) -> pd.DataFrame:
        """
        Recommends tracks based on an arbitrary target audio vector (e.g., high energy + high valence).
        """
        vec = np.array([target_features.get(f, 0.5) for f in NORM_AUDIO_FEATURES], dtype=np.float32).reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(vec, n_neighbors=n_recommendations)

        results = []
        for idx, dist in zip(indices.flatten(), distances.flatten()):
            row = self.df.loc[idx]
            sim_score = max(0.0, 1.0 - float(dist))
            results.append({
                'track_index': int(idx),
                'Artist': row['Artist'],
                'Track': row['Track'],
                'similarity_score': round(sim_score, 4),
                'vibe_name': row.get('vibe_name', 'General'),
                'popularity_score': round(float(row.get('popularity_score', 0.0)), 4),
                'Url_spotify': row.get('Url_spotify', ''),
                'Url_youtube': row.get('Url_youtube', '')
            })
        return pd.DataFrame(results)

    def get_track_similarity(self, track_a: int, track_b: int) -> float:
        """Computes cosine similarity between two tracks."""
        vec_a = self.feature_matrix[track_a].reshape(1, -1)
        vec_b = self.feature_matrix[track_b].reshape(1, -1)
        return float(cosine_similarity(vec_a, vec_b)[0][0])
