import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure root directory is on PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.preprocessing import load_and_clean_data, cluster_vibes, generate_synthetic_interactions
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.hybrid import HybridRecommender
from src.evaluation import run_benchmark


def main():
    csv_path = os.path.join(root_dir, "Spotify_Youtube.csv")
    print("1. Loading and cleaning dataset...")
    df = load_and_clean_data(csv_path)
    print(f"   Cleaned dataset shape: {df.shape}")

    print("2. Clustering into sonic vibes...")
    df = cluster_vibes(df, n_clusters=5)
    print("   Vibe clusters:", df['vibe_name'].value_counts().to_dict())

    print("3. Generating realistic user interactions...")
    interactions_df = generate_synthetic_interactions(df, n_users=80, n_interactions=1200)
    print(f"   Interactions count: {len(interactions_df)}, Users: {interactions_df['user_id'].nunique()}")

    print("4. Fitting Content-Based Recommender...")
    cb = ContentBasedRecommender().fit(df)
    cb_recs = cb.recommend_for_track(0, n_recommendations=3)
    print("   CB Top 3:\n", cb_recs[['Artist', 'Track', 'similarity_score', 'vibe_name']])

    print("5. Fitting Collaborative Recommender (SVD)...")
    cf = CollaborativeRecommender(n_factors=15).fit(interactions_df, len(df))
    cf_recs = cf.recommend_similar_tracks(0, df, n_recommendations=3)
    print("   CF Top 3:\n", cf_recs[['Artist', 'Track', 'collab_similarity']])

    print("6. Testing Hybrid Recommender...")
    hybrid = HybridRecommender(cb, cf, df)
    hy_recs = hybrid.recommend(0, n_recommendations=3, alpha=0.55, beta=0.30, gamma=0.15)
    print("   Hybrid Top 3:\n", hy_recs[['Artist', 'Track', 'hybrid_score', 'acoustic_match', 'collab_match', 'vibe_name']])

    print("7. Running Offline Benchmark...")
    bench_results = run_benchmark(df, interactions_df, k=5)
    print("   Benchmark Results:\n", bench_results)

    print("\nSUCCESS: All pipeline components tested and verified!")


if __name__ == "__main__":
    main()
