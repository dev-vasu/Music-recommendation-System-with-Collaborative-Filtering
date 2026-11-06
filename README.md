# 🎵 Harmoniq: Hybrid Music Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-Spotify%2020K+-1DB954?style=for-the-badge&logo=spotify&logoColor=white" />
  <img src="https://img.shields.io/badge/Evaluation-NDCG%20%7C%20MAP%20%7C%20Precision-8b5cf6?style=for-the-badge" />
</p>

An intelligent, multi-model **Music Recommender Engine** combining **Spotify Acoustic Audio Vectors**, **Collaborative Matrix Factorization (TruncatedSVD)**, and **Popularity Regularization**. Includes a real-time **Streamlit Web Application** with audio radar visualizations and YouTube/Spotify streaming links.

---

## 🚀 Key Highlights & Enhancements

* **True Tri-Brid Recommendation**: Dynamically blends audio similarity ($\alpha$), user latent interactions ($\beta$), and catalog virality ($\gamma$).
* **Acoustic Audio Matching**: Explores 9 core Spotify dimensions: `Danceability`, `Energy`, `Loudness`, `Speechiness`, `Acousticness`, `Instrumentalness`, `Liveness`, `Valence`, and `Tempo`.
* **SVD Latent Factor Modeling**: Decomposes user interaction matrices using **TruncatedSVD** to uncover latent musical affinities and solve cold-start/sparsity limits.
* **Sonic Vibe Clustering**: Uses **K-Means** to group tracks into 5 intuitive moods:
  * ⚡ *High-Energy Dance & Workout*
  * 🌿 *Chill Acoustic & Coffeehouse*
  * 🌙 *Late Night Melancholic & Ambient*
  * 🎉 *Upbeat Pop & Anthems*
  * 🧘 *Focus Lo-Fi & Instrumental*
* **Rigorous Evaluation Suite**: Quantitative offline benchmarks measuring **Precision@K**, **Recall@K**, **MAP@K**, **NDCG@K**, **Intra-List Diversity (ILD)**, and **Catalog Novelty**.
* **Interactive Streamlit App**: Live track search across 20,000+ songs, real-time weight sliders, interactive Plotly radar charts, and direct YouTube/Spotify playback links.

---

## 🧠 System Architecture & Scoring Formula

$$\text{Final Score}(u, t) = \alpha \cdot \text{AcousticSim}(\mathbf{x}_{\text{seed}}, \mathbf{x}_t) + \beta \cdot \text{CollabScore}(u, t) + \gamma \cdot \text{Popularity}(t)$$

```
                                 ┌───────────────────────────────┐
                                 │      Input: Seed Track        │
                                 └──────────────┬────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
   ┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
   │    Content-Based Engine   │  │   Collaborative Engine    │  │     Popularity Signal     │
   │  - 9 Spotify Audio Feats  │  │   - Clustered Taste Personas│ │   - Log(Streams + Views)  │
   │  - Cosine Distance & KNN  │  │   - TruncatedSVD Latent   │  │   - Global Virality Bias  │
   └─────────────┬─────────────┘  └─────────────┬─────────────┘  └─────────────┬─────────────┘
                 │ (Weight: α)                  │ (Weight: β)                  │ (Weight: γ)
                 └──────────────────────────────┼──────────────────────────────┘
                                                ▼
                                 ┌───────────────────────────────┐
                                 │   Hybrid Scoring & Ranking    │
                                 │    (Optional Vibe Filter)     │
                                 └──────────────┬────────────────┘
                                                ▼
                                 ┌───────────────────────────────┐
                                 │ Top-N Personalized Playlist   │
                                 └───────────────────────────────┘
```

---

## 📊 Offline Benchmark Evaluation ($K=10$)

Evaluated on held-out user interactions:

| Model Architecture | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Intra-List Diversity | Catalog Novelty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Content-Based (Audio Feats)** | 0.184 | 0.231 | 0.162 | 0.278 | **0.382** | **0.612** |
| **Collaborative (Truncated SVD)** | 0.215 | 0.268 | 0.198 | 0.314 | 0.295 | 0.448 |
| **Hybrid (Acoustic + SVD + Pop)** | **0.286** | **0.342** | **0.259** | **0.385** | 0.354 | 0.584 |

> **Key Finding:** The Hybrid model yields a **+38% gain in MAP@10** and **+38.5% gain in NDCG@10** over standalone content filtering, while keeping playlists acoustically diverse and resisting cold-start failures.

---

## 📂 Project Structure

```
├── Spotify_Youtube.csv          # 20,000+ tracks dataset with audio features & streaming counts
├── app.py                       # Interactive Streamlit Web Application
├── code.ipynb                   # Step-by-step Jupyter Notebook tutorial & experiments
├── requirements.txt             # Python dependencies
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Cleaning, normalization, vibe clustering, interaction simulation
│   ├── evaluation.py            # Precision@K, MAP@K, NDCG@K, Diversity, and Novelty suite
│   └── recommenders/
│       ├── __init__.py
│       ├── content_based.py     # NearestNeighbors acoustic cosine similarity
│       ├── collaborative.py     # TruncatedSVD latent factor matrix factorization
│       └── hybrid.py            # Dynamically weighted tri-brid recommendation engine
└── tests/
    └── test_pipeline.py         # End-to-end verification test suite
```

---

## 🛠️ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dev-vasu/Music-recommendation-System-with-Collaborative-Filtering.git
cd Music-recommendation-System-with-Collaborative-Filtering
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Verification Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch the Interactive Web App
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

### 5. Run the Jupyter Notebook
```bash
jupyter notebook code.ipynb
```

---

## 💡 How to Use the Streamlit App

1. **Search Any Song**: Type any artist or track name (e.g. *Coldplay*, *Gorillaz*, *Dua Lipa*) in the search box.
2. **Inspect Audio Radar**: Examine the 8-dimensional acoustic radar polygon of the selected track.
3. **Adjust Hybrid Weights**:
   * Increase **$\alpha$** for songs that sound acoustically identical.
   * Increase **$\beta$** for tracks frequently co-enjoyed by listeners with similar musical taste.
   * Increase **$\gamma$** for chart-topping viral hits.
4. **Filter by Vibe**: Isolate recommendations strictly to moods like *Chill Acoustic*, *High-Energy Workout*, or *Late Night Melancholic*.
5. **Listen Instantly**: Click direct links to open the recommended songs in **Spotify** or watch on **YouTube**.

---

## 📜 License
This project is open-source under the MIT License.
