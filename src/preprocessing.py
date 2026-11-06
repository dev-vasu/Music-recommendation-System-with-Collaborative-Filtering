"""
Data Preprocessing, Feature Engineering, and Interaction Modeling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

AUDIO_FEATURES = [
    'Danceability',
    'Energy',
    'Loudness',
    'Speechiness',
    'Acousticness',
    'Instrumentalness',
    'Liveness',
    'Valence',
    'Tempo'
]

NORM_AUDIO_FEATURES = [f'norm_{col}' for col in AUDIO_FEATURES]

VIBE_LABELS = {
    0: "⚡ High-Energy Dance & Workout",
    1: "🌿 Chill Acoustic & Coffeehouse",
    2: "🌙 Late Night Melancholic & Ambient",
    3: "🎉 Upbeat Pop & Anthems",
    4: "🧘 Focus Lo-Fi & Instrumental"
}


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Loads Spotify_Youtube.csv, drops corrupted rows, normalizes audio features,
    and computes composite popularity metrics.
    """
    df = pd.read_csv(csv_path)

    # Required columns
    metadata_cols = ['Artist', 'Track', 'Album', 'Url_spotify', 'Url_youtube']
    numeric_cols = AUDIO_FEATURES + ['Views', 'Likes', 'Stream']

    # Keep columns that exist in the file
    available_metadata = [c for c in metadata_cols if c in df.columns]
    available_numeric = [c for c in numeric_cols if c in df.columns]

    df = df[available_metadata + available_numeric].dropna(subset=available_metadata[:2] + AUDIO_FEATURES).copy()
    
    # Fill remaining missing numeric values with median
    for col in ['Views', 'Likes', 'Stream']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Remove duplicate artist-track pairs to maintain index integrity
    df = df.drop_duplicates(subset=['Artist', 'Track']).reset_index(drop=True)

    # Add normalized audio features for distance computation (0 to 1)
    scaler = MinMaxScaler()
    df[NORM_AUDIO_FEATURES] = scaler.fit_transform(df[AUDIO_FEATURES])

    # Compute a logarithmic popularity score (0 to 1)
    stream_metric = df['Stream'] if 'Stream' in df.columns else df['Views']
    log_views = np.log1p(df['Views'])
    log_likes = np.log1p(df['Likes'])
    log_stream = np.log1p(stream_metric)
    
    raw_pop = 0.4 * log_stream + 0.4 * log_views + 0.2 * log_likes
    pop_min = raw_pop.min()
    pop_max = raw_pop.max()
    df['popularity_score'] = (raw_pop - pop_min) / (pop_max - pop_min + 1e-9)

    return df


def cluster_vibes(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Groups tracks into sonic 'Vibes' using KMeans clustering over acoustic features.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(df[NORM_AUDIO_FEATURES])
    df['vibe_cluster'] = cluster_ids
    df['vibe_name'] = df['vibe_cluster'].map(lambda cid: VIBE_LABELS.get(cid, f"Vibe {cid}"))
    return df


def generate_synthetic_interactions(
    df: pd.DataFrame,
    n_users: int = 600,
    n_interactions: int = 20000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic, clustered user interactions:
    Instead of uniform random noise, users have preferred vibes/artists,
    and song selection follows a Pareto (power-law) popularity distribution.
    """
    rng = np.random.default_rng(seed)

    if 'vibe_cluster' not in df.columns:
        df = cluster_vibes(df)

    n_tracks = len(df)
    vibe_clusters = df['vibe_cluster'].values

    # Pre-calculate track popularity weights (power law)
    pop_scores = df['popularity_score'].values + 0.05
    track_probs = pop_scores / pop_scores.sum()

    # Assign each user 1-2 favorite vibes
    user_fav_vibes = {
        u: rng.choice(5, size=rng.integers(1, 3), replace=False)
        for u in range(1, n_users + 1)
    }

    interactions = []
    seen = set()

    # Target interactions per user follows a gamma distribution
    user_interaction_counts = rng.gamma(shape=2.5, scale=12.0, size=n_users).astype(int) + 5

    for u in range(1, n_users + 1):
        target_count = min(user_interaction_counts[u - 1], 80)
        favs = user_fav_vibes[u]

        # 70% chance to pick from favorite vibes, 30% from global popularity
        for _ in range(target_count):
            if rng.random() < 0.7:
                vibe_pick = rng.choice(favs)
                candidates = np.where(vibe_clusters == vibe_pick)[0]
                if len(candidates) > 0:
                    cand_probs = pop_scores[candidates] / pop_scores[candidates].sum()
                    t_idx = rng.choice(candidates, p=cand_probs)
                else:
                    t_idx = rng.choice(n_tracks, p=track_probs)
            else:
                t_idx = rng.choice(n_tracks, p=track_probs)

            pair = (u, t_idx)
            if pair not in seen:
                seen.add(pair)
                # Interaction strength (1 to 5 stars or repeat count)
                rating = rng.choice([3, 4, 5], p=[0.25, 0.45, 0.30])
                interactions.append({
                    'user_id': u,
                    'track_index': t_idx,
                    'rating': rating
                })

            if len(interactions) >= n_interactions:
                break
        if len(interactions) >= n_interactions:
            break

    interactions_df = pd.DataFrame(interactions)
    return interactions_df
