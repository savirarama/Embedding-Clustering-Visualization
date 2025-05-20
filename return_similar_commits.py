import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import argparse

def load_data(csv_file_path, npy_file_path):
    """
    Loads commit data from a CSV file and embeddings from a .npy file.

    Args:
        csv_file_path (str): Path to the CSV file containing commit information.
                               It must have a column named 'commit' for commit hashes.
        npy_file_path (str): Path to the .npy file containing commit embeddings.
                             The order of embeddings must match the order of commits in the CSV.

    Returns:
        tuple: A pandas DataFrame with commit information and a NumPy array with embeddings.
               Returns (None, None) if loading fails or files don't exist.
    """
    # Check if files exist
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return None, None
    if not os.path.exists(npy_file_path):
        print(f"Error: NPY file not found at {npy_file_path}")
        return None, None

    try:
        # Load commit information
        df_commits = pd.read_csv(csv_file_path)
        if 'commit' not in df_commits.columns:
            print("Error: CSV file must have a 'commit' column for commit hashes.")
            return None, None

        # Load commit embeddings
        embeddings = np.load(npy_file_path)

        # Validate that the number of commits matches the number of embeddings
        if len(df_commits) != len(embeddings):
            print(f"Error: Number of commits in CSV ({len(df_commits)}) "
                  f"does not match number of embeddings in NPY file ({len(embeddings)}).")
            print("Please ensure the .npy file contains embeddings in the same order as the CSV.")
            return None, None
            
        return df_commits, embeddings
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None, None

def load_query_commits(json_path):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            query_info_list = []
            for entry in data:
                query_hashes = entry.get("induceCommitHashList", [])
                target_hashes = entry.get("fixCommitHashList", [])
                for query_hash in query_hashes:
                    query_info_list.append({
                        "query": query_hash,
                        "targets": target_hashes
                    })
            return query_info_list
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return []

def find_similar_commits(commit_input_hash, n, df_commits, all_embeddings, target_commit_to_rank=None):
    """
    Finds the n most similar commits to a given commit hash.

    Args:
        commit_input_hash (str): The hash of the input commit.
        n (int): The number of similar commits to find.
        df_commits (pd.DataFrame): DataFrame containing commit information with a 'commit' column.
        all_embeddings (np.ndarray): NumPy array of all commit embeddings.

    Returns:
        list: A list of the n most similar commit hashes. Returns an empty list on error.
    """
    # 1. Find the index and embedding of the input commit
    try:
        # Get the row corresponding to the commit_input_hash
        target_commit_series = df_commits[df_commits['commit'] == commit_input_hash]
        
        if target_commit_series.empty:
            print(f"Error: Commit hash '{commit_input_hash}' not found in the CSV file.")
            return []
        
        # Get the index (iloc) of this commit in the DataFrame.
        # This index corresponds to the row number in all_embeddings.
        # Assuming default integer index for df_commits after read_csv.
        # If df_commits has a non-standard index, this might need adjustment,
        # but .index[0] gets the label, and we need its positional equivalent.
        # A safer way if index is not guaranteed to be 0-based sequential:
        target_idx = target_commit_series.index.values[0] 
        # Find the positional index if the DataFrame index is not simple 0..N-1
        try:
            # Try to get positional index directly if index is simple
            positional_target_idx = df_commits.index.get_loc(target_idx)
        except TypeError:
            # If index is not unique or other issues, fall back to iterating
            # This is less efficient but robust if index is complex
            positional_target_idx = -1
            for i, r_idx in enumerate(df_commits.index):
                if r_idx == target_idx:
                    positional_target_idx = i
                    break
            if positional_target_idx == -1: # Should not happen if target_commit_series was found
                 print(f"Error: Could not determine positional index for commit '{commit_input_hash}'.")
                 return []


        target_embedding = all_embeddings[positional_target_idx]

    except KeyError: # This would be caught by the 'Commit' column check in load_data
        print("Error: 'commit' column not found in the CSV file.")
        return []
    except IndexError: # If target_commit_series.index.values[0] fails
        print(f"Error: Could not retrieve index for commit hash '{commit_input_hash}'.")
        return []
    except Exception as e:
        print(f"An error occurred while finding the input commit embedding: {e}")
        return []

    # 2. Calculate cosine similarity
    # Reshape target_embedding to be a 2D array (1, num_features) for cosine_similarity function
    target_embedding_reshaped = target_embedding.reshape(1, -1)
    
    # Calculate similarities between the target and all embeddings
    # cosine_similarity returns a 2D array, e.g., [[sim_to_emb0, sim_to_emb1, ...]]
    similarity_matrix = cosine_similarity(target_embedding_reshaped, all_embeddings)
    
    # Extract the 1D array of similarity scores
    similarity_scores = similarity_matrix[0]

    # 3. Store similarities with commit hashes, excluding the input commit itself
    commit_similarity_pairs = []
    for i, score in enumerate(similarity_scores):
        # Exclude the input commit itself (which will have similarity ~1.0)
        if i == positional_target_idx:
            continue
        
        # Get the commit hash using iloc for positional access
        commit_hash = df_commits.iloc[i]['commit']
        commit_similarity_pairs.append({'hash': commit_hash, 'similarity': score})

    # 4. Sort by similarity in descending order
    # Using a lambda function to specify sorting by the 'similarity' value
    commit_similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # 5. Return top N commit hashes
    # Ensure we don't try to get more items than available
    num_results = min(n, len(commit_similarity_pairs))
    top_n_commits = [pair['hash'] for pair in commit_similarity_pairs[:num_results]]

    # 6. Find rank of target_commit_to_rank (if specified)
    target_rank = None
    target_similarity_score = None
    if target_commit_to_rank is not None:
        for rank, pair in enumerate(commit_similarity_pairs, start=1):
            if pair['hash'] == target_commit_to_rank:
                target_rank = rank
                target_similarity_score = pair['similarity']
                break
    
    return top_n_commits, target_rank, target_similarity_score

# --- Main execution ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find similar commits based on embeddings.')
    parser.add_argument('--csv', type=str,
                      help='Path to the CSV file containing commit information')
    parser.add_argument('--npy', type=str, default='embedding/compressed_embeddings.npy',
                      help='Path to the NPY file containing commit embeddings')
    parser.add_argument('--query', type=str, default='data/sid.json',
                      help='Path to the JSON file containing query commits')
    parser.add_argument('--output-recommendations', type=str, 
                      help='Path to save recommendation results')
    parser.add_argument('--output-ranks', type=str,
                      help='Path to save ranking results')
    parser.add_argument('--num-similar', type=int, default=10,
                      help='Number of top similar commits to retrieve')

    args = parser.parse_args()

    # --- Configuration ---
    csv_file_path = args.csv
    npy_file_path = args.npy
    query_json_path = args.query
    save_results_path = args.output_recommendations
    save_ranks_path = args.output_ranks
    num_similar_commits = args.num_similar

    print("Loading commit data...")
    df_all_commits, all_commit_embeddings = load_data(csv_file_path, npy_file_path)

    print(f"Loading query commits from '{query_json_path}'...")
    query_entries = load_query_commits(query_json_path)


    if df_all_commits is not None and all_commit_embeddings is not None and query_entries:
        # Initialize results
        rank_results = []
        recommendation_results = []

        all_true_labels = []
        all_pred_labels = []
        all_correct_predictions = 0
        total_queries = len(query_entries)

        # Process each query entry
        for entry in query_entries:
            query_hash = entry["query"]
            target_hashes = entry["targets"]

            print(f"\nProcessing query commit: {query_hash}")
            
            # --- Top-N similar commits for recommendation output ---
            similar_commits, _, _ = find_similar_commits(
                commit_input_hash=query_hash,
                n=num_similar_commits,
                df_commits=df_all_commits,
                all_embeddings=all_commit_embeddings
            )

            recommendation_results.append({
                "queryCommit": query_hash,
                "recommendedCommit": similar_commits
            })

            # --- Accuracy Calculation (run once per query) ---
            if any(target_hash in similar_commits for target_hash in target_hashes):
                all_correct_predictions += 1

            # --- Target commit ranking output ---
            for target_hash in target_hashes:
                _, rank, target_similarity_score = find_similar_commits(
                    commit_input_hash=query_hash,
                    n=num_similar_commits,
                    df_commits=df_all_commits,
                    all_embeddings=all_commit_embeddings,
                    target_commit_to_rank=target_hash
                )

                rank_results.append({
                    "query_commit": query_hash,
                    "target_commit": target_hash,
                    "rank": rank,
                    "similarity_score": float(target_similarity_score) if target_similarity_score is not None else None
                })

                # Binary ground truth: which top-N recommendations are actually relevant
                true_labels = [1 if commit in target_hashes else 0 for commit in similar_commits]
                pred_labels = [1] * len(similar_commits)  # predicted all as relevant (top-N)

                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)

                print(f"→ Target commit '{target_hash}' ranked #{rank} for query '{query_hash}'.")


        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='micro'
        )
        accuracy = all_correct_predictions / total_queries


        # Print metrics
        print("\nMetrics for all queries:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Save both outputs
        os.makedirs("recommendation_results", exist_ok=True)

        with open(save_results_path, "w") as f:
            json.dump(recommendation_results, f, indent=4)
            print(f"\n✔️ Recommendation results written to {save_results_path}")

        with open(save_ranks_path, "w") as f:
            json.dump(rank_results, f, indent=4)
            print(f"✔️ Ranking results written to {save_ranks_path}")
