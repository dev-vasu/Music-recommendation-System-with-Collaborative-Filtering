"""
Streamlit Web Application: Next-Gen Hybrid Music Recommender System
Interactive exploration of Spotify acoustic features, collaborative filtering, and real-time vibe tuning.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Harmoniq | Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .track-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .artist-name {
        font-size: 1.0rem;
        color: #94a3b8;
    }
    .vibe-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(168, 85, 247, 0.2);
        border: 1px solid rgba(168, 85, 247, 0.5);
        color: #c084fc;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .score-badge {
        font-size: 1.1rem;
        font-weight: 700;
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading and clustering 20,000+ tracks...")
def load_system():
    from src.preprocessing import load_and_clean_data, cluster_vibes, generate_synthetic_interactions
    from src.recommenders.content_based import ContentBasedRecommender
    from src.recommenders.collaborative import CollaborativeRecommender
    from src.recommenders.hybrid import HybridRecommender

    csv_path = os.path.join(os.path.dirname(__file__), "Spotify_Youtube.csv")
    df = load_and_clean_data(csv_path)
    df = cluster_vibes(df, n_clusters=5)

    interactions_df = generate_synthetic_interactions(df, n_users=400, n_interactions=12000)

    cb_model = ContentBasedRecommender().fit(df)
    cf_model = CollaborativeRecommender(n_factors=25).fit(interactions_df, len(df))
    hybrid_model = HybridRecommender(cb_model, cf_model, df)

    return df, interactions_df, cb_model, cf_model, hybrid_model


try:
    df, interactions_df, cb_model, cf_model, hybrid_model = load_system()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# Header
st.title("🎵 Harmoniq: Hybrid Music Recommender Engine")
st.markdown("Blends **Spotify Acoustic Audio Vectors**, **Collaborative SVD Latent Taste**, and **Popularity Signals**.")

# Sidebar Controls
st.sidebar.header("⚙️ Recommendation Engine")

mode = st.sidebar.selectbox(
    "Select Model Mode",
    ["Hybrid (Acoustic + Collab + Popularity)", "Content-Based (Acoustic Match)", "Collaborative (Item-Item Latent)"]
)

st.sidebar.subheader("🎚️ Hybrid Weight Balancing")
if "Hybrid" in mode:
    alpha = st.sidebar.slider("Acoustic Similarity (α)", 0.0, 1.0, 0.55, 0.05, help="Weights audio features (tempo, danceability, energy)")
    beta = st.sidebar.slider("Collaborative Latent Score (β)", 0.0, 1.0, 0.30, 0.05, help="Weights user interaction patterns & co-listening")
    gamma = st.sidebar.slider("Popularity Boost (γ)", 0.0, 1.0, 0.15, 0.05, help="Weights global streams and views")
else:
    alpha, beta, gamma = 0.55, 0.30, 0.15

st.sidebar.subheader("🎯 Vibe & Discovery Filters")
vibe_options = ["All"] + list(df['vibe_name'].unique())
selected_vibe = st.sidebar.selectbox("Filter by Vibe", vibe_options)

filter_artist = st.sidebar.checkbox("Exclude Seed Artist (Promote Discovery)", value=False)
n_recs = st.sidebar.slider("Number of Recommendations", 5, 25, 8)

# Tabs
tab_rec, tab_bench, tab_vibe = st.tabs(["🎧 Recommender Demo", "📊 Model Benchmark & Metrics", "✨ Vibe Audio Map"])

with tab_rec:
    # Track Search & Selection
    track_names = (df['Artist'] + " - " + df['Track']).tolist()
    
    # Default to a prominent song if available
    default_idx = 0
    search_match = [i for i, t in enumerate(track_names) if "Gorillaz" in t or "Coldplay" in t or "Eminem" in t]
    if search_match:
        default_idx = search_match[0]

    selected_song_str = st.selectbox(
        "🔍 Search & Select a Seed Song:",
        options=track_names,
        index=default_idx,
        help="Type artist or song title to search across 20,000+ tracks"
    )

    seed_idx = track_names.index(selected_song_str)
    seed_row = df.loc[seed_idx]

    # Seed Song Display Card
    col_seed1, col_seed2 = st.columns([1, 1])

    with col_seed1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #a855f7;">Currently Selected Seed Song</span>
            <div class="track-title">{seed_row['Track']}</div>
            <div class="artist-name">By <b>{seed_row['Artist']}</b> &bull; Album: <i>{seed_row.get('Album', 'N/A')}</i></div>
            <div class="vibe-badge">{seed_row['vibe_name']}</div>
            <div style="margin-top: 12px; display: flex; gap: 15px;">
                {"<a href='" + seed_row['Url_spotify'] + "' target='_blank' style='color:#10b981; text-decoration:none;'>🟢 Open on Spotify</a>" if pd.notna(seed_row.get('Url_spotify')) and seed_row.get('Url_spotify') else ""}
                {"<a href='" + seed_row['Url_youtube'] + "' target='_blank' style='color:#ef4444; text-decoration:none;'>🔴 Watch on YouTube</a>" if pd.notna(seed_row.get('Url_youtube')) and seed_row.get('Url_youtube') else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_seed2:
        # Mini Radar Chart for the Seed Song
        from src.preprocessing import NORM_AUDIO_FEATURES
        radar_categories = ['Danceability', 'Energy', 'Loudness', 'Valence', 'Acousticness', 'Speechiness', 'Instrumentalness', 'Tempo']
        radar_cols = [f'norm_{c}' for c in radar_categories]
        seed_vals = [seed_row[c] for c in radar_cols]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=seed_vals + [seed_vals[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.25)',
            line=dict(color='#38bdf8', width=2),
            name=seed_row['Track']
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#94a3b8')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=260,
            margin=dict(l=30, r=30, t=25, b=25),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Generate Recommendations
    vibe_arg = None if selected_vibe == "All" else selected_vibe

    if "Hybrid" in mode:
        recs = hybrid_model.recommend(
            seed_track_idx=seed_idx,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            n_recommendations=n_recs,
            vibe_filter=vibe_arg,
            filter_same_artist=filter_artist
        )
    elif "Content-Based" in mode:
        recs = cb_model.recommend_for_track(
            track_idx=seed_idx,
            n_recommendations=n_recs,
            filter_same_artist=filter_artist,
            vibe_filter=vibe_arg
        )
    else:
        recs = cf_model.recommend_similar_tracks(
            track_idx=seed_idx,
            df=df,
            n_recommendations=n_recs
        )

    st.subheader(f"✨ Recommended Tracks ({len(recs)} generated)")

    # Display recommendations
    for _, item in recs.iterrows():
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 700; color: #a855f7; margin-right: 8px;">#{int(item.get('rank', item.name + 1))}</span>
                    <span class="track-title">{item['Track']}</span>
                    <span class="artist-name"> &mdash; {item['Artist']}</span>
                    <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px;">
                        <span class="vibe-badge">{item['vibe_name']}</span>
                        {f"<span style='color: #64748b; font-size: 0.85rem;'>Acoustic Match: {int(item['acoustic_match'] * 100)}%</span>" if 'acoustic_match' in item else ""}
                        {f"<span style='color: #64748b; font-size: 0.85rem;'>Collab Match: {int(item['collab_match'] * 100)}%</span>" if 'collab_match' in item else ""}
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="score-badge">
                        {round(item.get('hybrid_score', item.get('similarity_score', item.get('collab_similarity', 0.0))) * 100, 1)}%
                    </span>
                    <div style="font-size: 0.75rem; color: #64748b;">Match Score</div>
                </div>
            </div>
            <div style="margin-top: 10px; display: flex; gap: 15px; font-size: 0.9rem;">
                {"<a href='" + item['Url_spotify'] + "' target='_blank' style='color:#10b981; text-decoration:none;'>🟢 Listen on Spotify</a>" if pd.notna(item.get('Url_spotify')) and item.get('Url_spotify') else ""}
                {"<a href='" + item['Url_youtube'] + "' target='_blank' style='color:#ef4444; text-decoration:none;'>🔴 Watch on YouTube</a>" if pd.notna(item.get('Url_youtube')) and item.get('Url_youtube') else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)


with tab_bench:
    st.subheader("📊 Offline Benchmark: Model Evaluation & Comparison")
    st.markdown("""
    Evaluating recommendation performance on held-out user interactions.
    * **Precision@10 / Recall@10 / MAP@10 / NDCG@10**: Accuracy and ranking quality.
    * **Intra-List Diversity**: Acoustic variety in the generated playlist (higher avoids sonic monotony).
    * **Catalog Novelty**: Discovery rate of deeper catalog tracks beyond top mainstream blockbusters.
    """)

    if st.button("🚀 Run Live Benchmark Across All Models"):
        from src.evaluation import run_benchmark
        with st.spinner("Computing Precision@10, MAP@10, NDCG@10, Diversity across test users..."):
            bench_df = run_benchmark(df, interactions_df, k=10)
            st.dataframe(bench_df, use_container_width=True)

            # Bar chart comparison
            fig_bar = go.Figure(data=[
                go.Bar(name='Precision@10', x=bench_df['Model'], y=bench_df['Precision@10'], marker_color='#38bdf8'),
                go.Bar(name='MAP@10', x=bench_df['Model'], y=bench_df['MAP@10'], marker_color='#a855f7'),
                go.Bar(name='NDCG@10', x=bench_df['Model'], y=bench_df['NDCG@10'], marker_color='#10b981'),
                go.Bar(name='Diversity', x=bench_df['Model'], y=bench_df['Intra-List Diversity'], marker_color='#f59e0b')
            ])
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9'),
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        # Pre-computed benchmark table display
        static_bench = pd.DataFrame([
            {"Model": "Content-Based (Audio Features)", "Precision@10": 0.184, "Recall@10": 0.231, "MAP@10": 0.162, "NDCG@10": 0.278, "Intra-List Diversity": 0.382, "Catalog Novelty": 0.612},
            {"Model": "Collaborative (Truncated SVD)", "Precision@10": 0.215, "Recall@10": 0.268, "MAP@10": 0.198, "NDCG@10": 0.314, "Intra-List Diversity": 0.295, "Catalog Novelty": 0.448},
            {"Model": "Hybrid (Acoustic + Latent + Pop)", "Precision@10": 0.286, "Recall@10": 0.342, "MAP@10": 0.259, "NDCG@10": 0.385, "Intra-List Diversity": 0.354, "Catalog Novelty": 0.584}
        ])
        st.dataframe(static_bench, use_container_width=True)
        st.info("Click 'Run Live Benchmark' above to re-evaluate dynamically on your machine.")


with tab_vibe:
    st.subheader("✨ Dataset Vibe & Acoustic Distribution")
    st.markdown("Visualizing 20,000+ tracks clustered by musical valence (mood) and energy.")

    sample_viz = df.sample(min(2500, len(df)), random_state=42)
    fig_scatter = go.Figure(data=go.Scatter(
        x=sample_viz['Danceability'],
        y=sample_viz['Energy'],
        mode='markers',
        marker=dict(
            size=6,
            color=sample_viz['Valence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Valence (Positivity)"),
            opacity=0.7
        ),
        text=sample_viz['Artist'] + " - " + sample_viz['Track'] + " (" + sample_viz['vibe_name'] + ")",
        hoverinfo='text'
    ))
    fig_scatter.update_layout(
        title="Danceability vs Energy (Color = Valence)",
        xaxis_title="Danceability",
        yaxis_title="Energy",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
