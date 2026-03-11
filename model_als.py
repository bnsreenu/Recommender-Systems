"""
Instacart Recommender System - ALS Model (Tutorial 2)

This module implements Alternating Least Squares (ALS) matrix factorization
for collaborative filtering using the implicit library.

ALS is a classical approach that learns latent factors for users and items
by decomposing the user-item interaction matrix. It works well with implicit
feedback data (purchases, clicks) where we only observe positive interactions.

Key advantages:
- Fast training (can use GPU)
- Scalable to millions of users/items
- Produces interpretable embeddings
- Proven technique used in production systems

The model learns:
- User embeddings: each user → 64-dim vector
- Item embeddings: each product → 64-dim vector
- Recommendation score = dot product of user and item vectors
"""

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import warnings
warnings.filterwarnings('ignore')


def train_als(train_matrix, factors=64, regularization=0.01, iterations=15, 
              use_gpu=True, alpha=40):
    """
    Train ALS model on user-item interaction matrix.
    
    The model learns latent factors that capture user preferences and product
    characteristics. Users with similar purchase histories get similar embeddings,
    and products that are frequently bought together get similar embeddings.
    
    Parameters:
    -----------
    train_matrix : scipy sparse matrix (users × items)
        Training interaction matrix. Non-zero entries indicate purchases.
        Values can be purchase counts (implicit confidence) or binary (0/1).
    
    factors : int, default=64
        Dimensionality of the latent factors (embedding size).
        Higher = more expressive but slower and may overfit.
        Common values: 32, 64, 128
    
    regularization : float, default=0.01
        L2 regularization strength to prevent overfitting.
        Higher values = more regularization = simpler model.
        Typical range: 0.001 to 0.1
    
    iterations : int, default=15
        Number of ALS iterations.
        Each iteration alternates between updating user and item factors.
        Usually converges in 10-20 iterations.
    
    use_gpu : bool, default=True
        Use GPU acceleration if available (requires cupy).
        Falls back to CPU if GPU not available.
    
    alpha : float, default=40
        Confidence scaling for implicit feedback.
        Transforms raw counts to confidence: confidence = 1 + alpha * count
        Higher alpha = more weight on observed interactions.
    
    Returns:
    --------
    model : AlternatingLeastSquares
        Trained ALS model with learned user and item factors
    
    Notes:
    ------
    The loss function being minimized is:
    L = Σ c_ui * (p_ui - user_i · item_j)² + λ(||user_i||² + ||item_j||²)
    
    where:
    - c_ui is confidence (1 + alpha * r_ui for observed, 1 for unobserved)
    - p_ui is preference (1 for observed, 0 for unobserved)
    - λ is regularization
    """
    
    print("\n" + "="*70)
    print("TRAINING ALS MODEL")
    print("="*70)
    
    print(f"\nModel Configuration:")
    print(f"  Embedding dimension: {factors}")
    print(f"  Regularization: {regularization}")
    print(f"  Iterations: {iterations}")
    print(f"  Alpha (confidence): {alpha}")
    print(f"  Use GPU: {use_gpu}")
    
    print(f"\nTraining Data:")
    print(f"  Shape: {train_matrix.shape[0]:,} users × {train_matrix.shape[1]:,} items")
    print(f"  Non-zero entries: {train_matrix.nnz:,}")
    print(f"  Sparsity: {(1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])):.2%}")
    
    # Apply BM25 weighting to reduce popularity bias
    # This down-weights popular items so they don't dominate recommendations
    print("\nApplying BM25 weighting to reduce popularity bias...")
    train_matrix_weighted = bm25_weight(train_matrix, K1=100, B=0.8)
    
    # Convert to CSR format (users × items)
    # Note: implicit library works with both formats, we use users × items
    train_matrix_csr = train_matrix_weighted.tocsr()
    
    # Initialize model
    print("\nInitializing ALS model...")
    model = AlternatingLeastSquares(
        factors=factors,  #64
        regularization=regularization,  #0.01
        iterations=iterations,  #15
        use_gpu=use_gpu,  #Currently False
        random_state=42
    )
    
    # Train model
    print("\nTraining (this may take 1-2 minutes)...")
    model.fit(train_matrix_csr, show_progress=True)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Model now contains:
    # - model.user_factors: (n_users × factors) matrix
    # - model.item_factors: (n_items × factors) matrix
    
    return model


def predict_als(model, train_matrix, user_ids=None, k=10, 
                filter_already_purchased=True):
    """
    Generate top-K recommendations for users using trained ALS model.
    
    For each user, computes scores for all items using dot product of
    user embedding and item embeddings. Returns top-K items with highest scores.
    
    Parameters:
    -----------
    model : AlternatingLeastSquares
        Trained ALS model
    
    train_matrix : scipy sparse matrix (users × items)
        Training data used to filter out already purchased items
    
    user_ids : list or None
        User indices to generate recommendations for.
        If None, generates for all users.
    
    k : int, default=10
        Number of recommendations per user
    
    filter_already_purchased : bool, default=True
        If True, exclude items the user already purchased in training.
        Typically True for evaluation, False for actual deployment.
    
    Returns:
    --------
    predictions : dict
        {user_idx: [list of k product indices]}
        Products are ordered by predicted score (highest first)
    
    scores : dict
        {user_idx: [list of k scores]}
        Corresponding confidence scores for each recommendation
    
    Notes:
    ------
    Recommendation score for user u and item i:
    score(u, i) = user_embedding[u] · item_embedding[i]
    
    Higher score = more likely to purchase
    """
    
    print("\nGenerating recommendations...")
    
    # Default to all users
    if user_ids is None:
        user_ids = list(range(train_matrix.shape[0]))
    
    predictions = {}
    scores_dict = {}
    
    # Convert to CSR format (users × items)
    train_matrix_csr = train_matrix.tocsr()
    
    print(f"  Predicting for {len(user_ids):,} users...")
    
    # Use batch recommendation for efficiency
    # recommend returns (item_ids, scores) for batch of users
    batch_items, batch_scores = model.recommend(
        userid=user_ids,
        user_items=train_matrix_csr,
        N=k,
        filter_already_liked_items=filter_already_purchased
    )
    
    # Convert batch results to dictionary format
    for i, user_idx in enumerate(user_ids):
        predictions[user_idx] = batch_items[i].tolist()
        scores_dict[user_idx] = batch_scores[i].tolist()
    
    print(f"  Generated {k} recommendations per user")
    
    return predictions, scores_dict


def get_similar_items(model, item_ids, n=10):
    """
    Find items similar to given items.
    
    Uses item embeddings to find products that are frequently purchased
    together with the query items. Useful for "customers also bought"
    style recommendations.
    
    Parameters:
    -----------
    model : AlternatingLeastSquares
        Trained ALS model
    
    item_ids : list
        Product indices to find similar items for
    
    n : int, default=10
        Number of similar items to return per query item
    
    Returns:
    --------
    dict : {item_id: [(similar_item_id, score), ...]}
    """
    
    similar_items = {}
    
    for item_id in item_ids:
        # Find similar items using cosine similarity of embeddings
        items, scores = model.similar_items(itemid=item_id, N=n+1)
        
        # Remove the query item itself (first result)
        similar_items[item_id] = list(zip(items[1:], scores[1:]))
    
    return similar_items


def get_item_embeddings(model):
    """
    Extract item embedding matrix from trained model.
    
    These embeddings capture product characteristics learned from purchase patterns.
    Similar products (e.g., different brands of milk) have similar embeddings.
    
    Parameters:
    -----------
    model : AlternatingLeastSquares
        Trained ALS model
    
    Returns:
    --------
    embeddings : numpy array, shape (n_items, factors)
        Item factor matrix where each row is a product's embedding vector
    
    Notes:
    ------
    Can be used for:
    - Visualization (PCA/t-SNE to 2D)
    - Item-to-item similarity
    - Product clustering
    - Cold start for new products (use content features to predict embedding)
    """
    
    return model.item_factors


def get_user_embeddings(model):
    """
    Extract user embedding matrix from trained model.
    
    These embeddings capture user preferences learned from purchase history.
    Users with similar tastes have similar embeddings.
    
    Parameters:
    -----------
    model : AlternatingLeastSquares
        Trained ALS model
    
    Returns:
    --------
    embeddings : numpy array, shape (n_users, factors)
        User factor matrix where each row is a user's embedding vector
    
    Notes:
    ------
    Can be used for:
    - User segmentation
    - Finding similar users
    - Cold start analysis (understand what makes a user predictable)
    """
    
    return model.user_factors


def explain_recommendations(model, user_idx, item_idx, train_matrix, 
                           product_info, top_n=5):
    """
    Explain why an item was recommended to a user.
    
    Shows which products the user previously purchased that led to
    this recommendation (based on embedding similarity).
    
    Parameters:
    -----------
    model : AlternatingLeastSquares
        Trained ALS model
    
    user_idx : int
        User index
    
    item_idx : int
        Recommended item index
    
    train_matrix : scipy sparse matrix
        Training data showing user's purchase history
    
    product_info : DataFrame
        Product metadata
    
    top_n : int
        Number of explanatory items to return
    
    Returns:
    --------
    list : [(product_name, contribution_score), ...]
    
    Notes:
    ------
    Contribution score = user_history_item · recommended_item
    Items with highest contribution most influenced this recommendation.
    """
    
    # Get user's purchase history
    user_items = train_matrix[user_idx].nonzero()[1]
    
    if len(user_items) == 0:
        return []
    
    # Get embeddings
    user_embedding = model.user_factors[user_idx]
    item_embedding = model.item_factors[item_idx]
    history_embeddings = model.item_factors[user_items]
    
    # Calculate contribution of each history item
    contributions = history_embeddings.dot(item_embedding)
    
    # Get top contributing items
    top_indices = np.argsort(contributions)[::-1][:top_n]
    
    explanations = []
    for idx in top_indices:
        history_item_idx = user_items[idx]
        contrib_score = contributions[idx]
        
        # Get product name
        prod_row = product_info[product_info['product_idx'] == history_item_idx]
        if not prod_row.empty:
            product_name = prod_row['product_name'].values[0]
            explanations.append((product_name, contrib_score))
    
    return explanations


def get_model_statistics(model, train_matrix):
    """
    Calculate statistics about the trained model.
    
    Useful for understanding model complexity and potential issues.
    
    Parameters:
    -----------
    model : AlternatingLeastSquares
        Trained ALS model
    
    train_matrix : scipy sparse matrix
        Training data
    
    Returns:
    --------
    dict : Model statistics
    """
    
    stats = {}
    
    # Embedding statistics
    user_factors = model.user_factors
    item_factors = model.item_factors
    
    stats['n_users'] = user_factors.shape[0]
    stats['n_items'] = item_factors.shape[0]
    stats['embedding_dim'] = user_factors.shape[1]
    
    # Check for potential issues
    stats['user_embedding_norm_mean'] = np.linalg.norm(user_factors, axis=1).mean()
    stats['item_embedding_norm_mean'] = np.linalg.norm(item_factors, axis=1).mean()
    
    # Sparsity
    stats['train_sparsity'] = 1 - (train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]))
    
    # Average interactions per user
    stats['avg_items_per_user'] = train_matrix.nnz / train_matrix.shape[0]
    
    return stats


def save_als_model(model, filepath):
    """
    Save trained ALS model to disk.
    
    Parameters:
    -----------
    model : AlternatingLeastSquares
        Trained model
    
    filepath : str
        Path to save model (will create .npz file)
    """
    
    import pickle
    
    # Save model using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nALS model saved to: {filepath}")


def load_als_model(filepath):
    """
    Load saved ALS model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to saved model file
    
    Returns:
    --------
    model : AlternatingLeastSquares
        Loaded model
    """
    
    import pickle
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"\nALS model loaded from: {filepath}")
    
    return model


# =============================================================================
# 
# =============================================================================

if __name__ == "__main__":
    """
    Example showing how to train and use the ALS model.
    This is just for testing - actual usage is in run_tutorial_2.py
    """
    
    print("ALS Model Module - Example Usage")
    print("="*70)
    print("\nThis module contains functions for training ALS models.")
    print("For full tutorial, run: python run_tutorial_2.py")
    print("\nKey functions:")
    print("  - train_als(): Train the model")
    print("  - predict_als(): Generate recommendations")
    print("  - get_item_embeddings(): Extract embeddings for visualization")
    print("  - explain_recommendations(): Understand why items were recommended")