# Embedding Clustering and Visualization

This project compresses and visualizes commit message embeddings using a Multi-Layer Perceptron (MLP) neural network. It provides tools for both visualization and similarity-based commit recommendation.

## Project Structure

```
.
├── data/
│   ├── test.csv           # Commit data
│   └── sid.json           # Query commit information
├── embedding/
│   ├── embeddings.npy     # Original embeddings
│   └── compressed_embeddings.npy  # MLP-compressed embeddings
├── recommendation_results/
│   ├── compressed_recommendation_results.json  # Similar commit recommendations based on original embeddings
│   ├── compressed_rank_results.json            # Ranking results based on original embeddings
│   ├── initial_recommendation_results.json  # Similar commit recommendations based on original embeddings
│   └── initial_rank_results.json           # Ranking results based on original embeddings
├── mlp_cluster_visualization.ipynb  # Visualization notebook
└── return_similar_commits.py        # Similarity search script
```

## Features

1. **MLP-based Embedding Compression**
   - Compresses 768-dimensional embeddings to 64 dimensions
   - Uses a three-layer MLP with batch normalization and ReLU activation
   - Architecture: 768 → 512 → 256 → 64

2. **Visualization**
   - t-SNE visualization of both original and compressed embeddings
   - Side-by-side comparison of clustering patterns
   - Interactive Jupyter notebook interface

3. **Similarity Search**
   - Cosine similarity-based commit recommendation
   - Ranking of similar commits
   - Performance metrics calculation (Precision, Recall, F1, Accuracy)

## Requirements

Python 3.8+ is required. Install dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning utilities
- jupyter: Interactive notebook interface
- matplotlib: Visualization
- seaborn: Statistical data visualization
- torch: PyTorch for MLP implementation

## Usage

### 1. Visualization

Run the Jupyter notebook to visualize the embeddings.


The notebook will:
- Load the original embeddings
- Compress them using the MLP
- Generate t-SNE visualizations
- Save the compressed embeddings

### 2. Similarity Search

Run the similarity search script:

```bash
python return_similar_commits.py --csv data/test.csv --npy embedding/compressed_embeddings.npy --query data/sid.json --output-recommendations recommendation_results/compressed_recommendation_results.json --output-ranks recommendation_results/compressed_rank_results.json --num-similar 10
```

This will:
- Load the compressed/initial embeddings
- Process query commits from sid.json
- Generate recommendations and rankings
- Calculate performance metrics
- Save results to JSON files

## Output Files

1. **Compressed Embeddings**
   - 64-dimensional compressed vectors

2. **Recommendation Results**
   - Contains top-N similar commits for each query

3. **Ranking Results**
   - Contains ranking information and similarity scores

## Performance Metrics

The system calculates and reports:
- Precision
- Recall
- F1 Score
- Accuracy

These metrics are calculated based on the ability to correctly identify relevant commits in the top-N recommendations.
