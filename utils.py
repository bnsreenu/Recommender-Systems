"""
Instacart Recommender System - Utility Functions

This module contains all reusable functions for data loading, preprocessing,
evaluation, and visualization. These functions are shared across both the
classical ALS model and the neural collaborative filtering model.

Functions:
- Data loading and preprocessing
- Train/test split generation
- NDCG and other metric calculations
- Recommendation evaluation
- Cold start handling
- Result visualization
"""

import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from pathlib import Path
from sklearn.metrics import ndcg_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(raw_data_path, processed_data_path, 
                          n_users=20000, n_products=800, 
                          min_orders=5, max_orders=20):
    """
    Load raw Instacart data and create a subset for tutorials.
    
    This creates a manageable subset with ~20K active users and ~800 popular products.
    The subset maintains realistic sparsity (~1%) while being computationally tractable.
    
    Parameters:
    -----------
    raw_data_path : str
        Path to folder containing raw Instacart CSV files
    processed_data_path : str
        Path where processed files will be saved
    n_users : int
        Target number of users in subset
    n_products : int
        Target number of products in subset
    min_orders : int
        Minimum number of orders per user to include
    max_orders : int
        Maximum number of orders per user to include
    
    Files created:
    --------------
    - user_item_matrix.npz : Full interaction matrix
    - train_matrix.npz : Training split
    - test_matrix.npz : Test split
    - user_mapping.pkl : user_id to matrix index mapping
    - item_mapping.pkl : product_id to matrix index mapping
    - product_info.csv : Product metadata
    - user_features.csv : User-level features for neural model
    """
    
    print("="*70)
    print("INSTACART DATA PREPARATION")
    print("="*70)
    
    # Create output directory
    Path(processed_data_path).mkdir(parents=True, exist_ok=True)
    
    # Load raw data files
    print("\nLoading raw Instacart data...")
    orders = pd.read_csv(f"{raw_data_path}/orders.csv")
    order_products = pd.read_csv(f"{raw_data_path}/order_products__train.csv")
    products = pd.read_csv(f"{raw_data_path}/products.csv")
    aisles = pd.read_csv(f"{raw_data_path}/aisles.csv")
    departments = pd.read_csv(f"{raw_data_path}/departments.csv")
    
    print(f"  Orders: {len(orders):,}")
    print(f"  Order-Products: {len(order_products):,}")
    print(f"  Products: {len(products):,}")
    
    # Merge orders with products first
    print(f"\nMerging orders with products...")
    order_data = orders.merge(order_products, on='order_id')
    
    print(f"  Total interactions: {len(order_data):,}")
    
    # Strategy: Instacart data has 1 order per user in train set
    # So we filter by number of PRODUCTS purchased, not number of orders
    print(f"\nFinding users with sufficient purchase history...")
    
    # Count products per user (not orders)
    user_product_counts = order_data.groupby('user_id')['product_id'].count()
    
    # Users should have bought at least min_orders products (reusing parameter name)
    min_products = max(5, min_orders)  # At least 5 products
    max_products_per_user = 50  # Cap at 50 products per order
    
    valid_users = user_product_counts[
        (user_product_counts >= min_products) & 
        (user_product_counts <= max_products_per_user)
    ].index
    
    print(f"  Found {len(valid_users):,} users with {min_products}-{max_products_per_user} products")
    
    if len(valid_users) < n_users:
        print(f"  WARNING: Only found {len(valid_users):,} users (target: {n_users:,})")
        print(f"  Using all available users")
        selected_users = valid_users
    else:
        # Select subset of users
        selected_users = np.random.choice(valid_users, n_users, replace=False)
        print(f"  Selected {len(selected_users):,} users")
    
    # Filter to selected users
    order_data = order_data[order_data['user_id'].isin(selected_users)]
    
    # Now select top products from this filtered set
    print(f"\nSelecting top {n_products} products...")
    product_counts = order_data['product_id'].value_counts()
    top_products = product_counts.head(n_products).index
    
    # Filter to top products
    order_data = order_data[order_data['product_id'].isin(top_products)]
    
    print(f"\n  Final dataset:")
    print(f"    Products: {len(order_data['product_id'].unique())}")
    print(f"    Users: {len(order_data['user_id'].unique()):,}")
    print(f"    Interactions: {len(order_data):,}")
    
    filtered_orders = orders[orders['user_id'].isin(selected_users)]
    
    print(f"  Final dataset: {len(order_data):,} purchases")
    print(f"  Users: {order_data['user_id'].nunique():,}")
    print(f"  Products: {order_data['product_id'].nunique():,}")
    
    # Create user and item mappings
    print("\nCreating index mappings...")
    unique_users = sorted(order_data['user_id'].unique())
    unique_products = sorted(order_data['product_id'].unique())
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    
    product_to_idx = {prod_id: idx for idx, prod_id in enumerate(unique_products)}
    idx_to_product = {idx: prod_id for prod_id, idx in product_to_idx.items()}
    
    # Map to indices
    order_data['user_idx'] = order_data['user_id'].map(user_to_idx)
    order_data['product_idx'] = order_data['product_id'].map(product_to_idx)
    
    # Build sparse interaction matrix
    print("\nBuilding user-item interaction matrix...")
    n_users_final = len(unique_users)
    n_products_final = len(unique_products)
    
    # Count interactions (number of times each user bought each product)
    interaction_counts = order_data.groupby(['user_idx', 'product_idx']).size()
    
    rows = interaction_counts.index.get_level_values(0)
    cols = interaction_counts.index.get_level_values(1)
    data = interaction_counts.values
    
    user_item_matrix = sp.csr_matrix(
        (data, (rows, cols)), 
        shape=(n_users_final, n_products_final)
    )
    
    sparsity = 1 - (user_item_matrix.nnz / (n_users_final * n_products_final))
    print(f"  Matrix shape: {user_item_matrix.shape}")
    print(f"  Non-zero entries: {user_item_matrix.nnz:,}")
    print(f"  Sparsity: {sparsity:.2%}")
    
    # Create train/test split
    print("\nCreating train/test split (80/20)...")
    train_matrix, test_matrix = create_train_test_split(
        order_data, user_to_idx, product_to_idx, 
        n_users_final, n_products_final
    )
    
    # Save matrices
    print("\nSaving processed data...")
    sp.save_npz(f"{processed_data_path}/user_item_matrix.npz", user_item_matrix)
    sp.save_npz(f"{processed_data_path}/train_matrix.npz", train_matrix)
    sp.save_npz(f"{processed_data_path}/test_matrix.npz", test_matrix)
    
    # Save mappings
    with open(f"{processed_data_path}/user_mapping.pkl", 'wb') as f:
        pickle.dump({'user_to_idx': user_to_idx, 'idx_to_user': idx_to_user}, f)
    
    with open(f"{processed_data_path}/item_mapping.pkl", 'wb') as f:
        pickle.dump({'product_to_idx': product_to_idx, 'idx_to_product': idx_to_product}, f)
    
    # Prepare and save product metadata
    print("\nPreparing product metadata...")
    product_info = products[products['product_id'].isin(top_products)].copy()
    product_info = product_info.merge(aisles, on='aisle_id')
    product_info = product_info.merge(departments, on='department_id')
    product_info['product_idx'] = product_info['product_id'].map(product_to_idx)
    product_info.to_csv(f"{processed_data_path}/product_info.csv", index=False)
    
    # Create user features for neural model
    print("\nCreating user features...")
    user_features = create_user_features(filtered_orders, order_data, user_to_idx)
    user_features.to_csv(f"{processed_data_path}/user_features.csv", index=False)
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"\nFiles saved to: {processed_data_path}/")
    print(f"  - user_item_matrix.npz ({n_users_final} × {n_products_final})")
    print(f"  - train_matrix.npz")
    print(f"  - test_matrix.npz")
    print(f"  - user_mapping.pkl")
    print(f"  - item_mapping.pkl")
    print(f"  - product_info.csv")
    print(f"  - user_features.csv")
    
    return {
        'n_users': n_users_final,
        'n_products': n_products_final,
        'n_interactions': user_item_matrix.nnz,
        'sparsity': sparsity
    }


def load_processed_data(processed_data_path):
    """
    Load previously prepared data from disk.
    
    This is the main function to call at the start of each tutorial.
    It loads the train/test split and all necessary mappings.
    
    Parameters:
    -----------
    processed_data_path : str
        Path to folder containing processed data files
    
    Returns:
    --------
    dict containing:
        - train_matrix : scipy sparse matrix (users × products)
        - test_matrix : scipy sparse matrix (users × products)
        - mappings : dict with user and product mappings
        - product_info : DataFrame with product metadata
        - user_features : DataFrame with user features
    """
    
    print("Loading processed data...")
    
    # Load matrices
    train_matrix = sp.load_npz(f"{processed_data_path}/train_matrix.npz")
    test_matrix = sp.load_npz(f"{processed_data_path}/test_matrix.npz")
    
    # Load mappings
    with open(f"{processed_data_path}/user_mapping.pkl", 'rb') as f:
        user_mapping = pickle.load(f)
    
    with open(f"{processed_data_path}/item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)
    
    # Load metadata
    product_info = pd.read_csv(f"{processed_data_path}/product_info.csv")
    user_features = pd.read_csv(f"{processed_data_path}/user_features.csv")
    
    print(f"  Train matrix: {train_matrix.shape}")
    print(f"  Test matrix: {test_matrix.shape}")
    print(f"  Products: {len(product_info)}")
    print(f"  Users: {len(user_features)}")
    
    return {
        'train_matrix': train_matrix,
        'test_matrix': test_matrix,
        'mappings': {
            'user_to_idx': user_mapping['user_to_idx'],
            'idx_to_user': user_mapping['idx_to_user'],
            'product_to_idx': item_mapping['product_to_idx'],
            'idx_to_product': item_mapping['idx_to_product']
        },
        'product_info': product_info,
        'user_features': user_features
    }


def create_train_test_split(order_data, user_to_idx, product_to_idx, 
                            n_users, n_products, test_ratio=0.2):
    """
    Create train/test split by holding out products from each user's order.
    
    Since Instacart train data has 1 order per user, we split the products
    within that order: 80% of products to train, 20% to test.
    
    Parameters:
    -----------
    order_data : DataFrame
        Order data with user_id, product_id columns
    user_to_idx : dict
        Mapping from user_id to matrix index
    product_to_idx : dict
        Mapping from product_id to matrix index
    n_users : int
        Number of users
    n_products : int
        Number of products
    test_ratio : float
        Fraction of products to hold out for testing
    
    Returns:
    --------
    train_matrix, test_matrix : scipy sparse matrices
    """
    
    train_rows, train_cols, train_data = [], [], []
    test_rows, test_cols, test_data = [], [], []
    
    # For each user, split their products into train/test
    for user_id, user_data in order_data.groupby('user_id'):
        user_idx = user_to_idx.get(user_id)
        if user_idx is None:
            continue
        
        # Get all products this user bought
        user_products = user_data['product_id'].values
        n_user_products = len(user_products)
        
        # Need at least 2 products to split
        if n_user_products < 2:
            # If only 1 product, put it in training
            product_idx = product_to_idx.get(user_products[0])
            if product_idx is not None:
                train_rows.append(user_idx)
                train_cols.append(product_idx)
                train_data.append(1)
            continue
        
        # Randomly shuffle and split
        shuffled_products = np.random.permutation(user_products)
        n_test = max(1, int(n_user_products * test_ratio))
        
        test_products = shuffled_products[:n_test]
        train_products = shuffled_products[n_test:]
        
        # Add to train set
        for product_id in train_products:
            product_idx = product_to_idx.get(product_id)
            if product_idx is not None:
                train_rows.append(user_idx)
                train_cols.append(product_idx)
                train_data.append(1)
        
        # Add to test set
        for product_id in test_products:
            product_idx = product_to_idx.get(product_id)
            if product_idx is not None:
                test_rows.append(user_idx)
                test_cols.append(product_idx)
                test_data.append(1)
    
    # Build sparse matrices
    train_matrix = sp.csr_matrix(
        (train_data, (train_rows, train_cols)),
        shape=(n_users, n_products)
    )
    
    test_matrix = sp.csr_matrix(
        (test_data, (test_rows, test_cols)),
        shape=(n_users, n_products)
    )
    
    # Convert to binary
    train_matrix = (train_matrix > 0).astype(int)
    test_matrix = (test_matrix > 0).astype(int)
    
    print(f"  Train: {train_matrix.nnz:,} interactions")
    print(f"  Test: {test_matrix.nnz:,} interactions")
    
    return train_matrix, test_matrix


def create_user_features(orders_df, order_data, user_to_idx):
    """
    Create user-level features for neural model.
    
    Features include:
    - Total number of orders
    - Average basket size
    - Favorite department
    - Days since first order
    - Purchase frequency patterns
    
    Parameters:
    -----------
    orders_df : DataFrame
        Order-level data
    order_data : DataFrame
        Order-product level data
    user_to_idx : dict
        User ID to index mapping
    
    Returns:
    --------
    DataFrame with user features
    """
    
    user_features = []
    
    for user_id in user_to_idx.keys():
        user_orders = orders_df[orders_df['user_id'] == user_id]
        user_products = order_data[order_data['user_id'] == user_id]
        
        # Basic stats
        n_orders = len(user_orders)
        avg_basket_size = len(user_products) / n_orders if n_orders > 0 else 0
        
        # Temporal features
        days_since_prior = user_orders['days_since_prior_order'].fillna(0)
        avg_days_between_orders = days_since_prior.mean()
        
        # Day of week preference
        dow_mode = user_orders['order_dow'].mode()
        favorite_dow = dow_mode.iloc[0] if len(dow_mode) > 0 else 0
        
        # Hour of day preference
        hour_mode = user_orders['order_hour_of_day'].mode()
        favorite_hour = hour_mode.iloc[0] if len(hour_mode) > 0 else 0
        
        user_features.append({
            'user_idx': user_to_idx[user_id],
            'user_id': user_id,
            'n_orders': n_orders,
            'avg_basket_size': avg_basket_size,
            'avg_days_between_orders': avg_days_between_orders,
            'favorite_dow': favorite_dow,
            'favorite_hour': favorite_hour
        })
    
    return pd.DataFrame(user_features)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_ndcg(predictions_dict, ground_truth_matrix, k_values=[5, 10]):
    """
    Calculate NDCG@K for multiple K values.
    
    This is the core evaluation metric connecting to Tutorial 1.
    NDCG measures how well the model ranks relevant items at the top.
    
    Parameters:
    -----------
    predictions_dict : dict
        {user_idx: [list of product_idx in ranked order]}
    ground_truth_matrix : scipy sparse matrix
        Binary matrix where 1 = user purchased product
    k_values : list
        K values to evaluate (e.g., [5, 10])
    
    Returns:
    --------
    dict : {f'ndcg@{k}': score} for each k
    """
    
    ndcg_scores = {f'ndcg@{k}': [] for k in k_values}
    
    n_products = ground_truth_matrix.shape[1]
    
    # OPTIMIZATION: Only process users who have predictions
    for user_idx, pred_products in predictions_dict.items():
        # Ground truth: which products did this user buy?
        true_products = ground_truth_matrix[user_idx].toarray().flatten()
        
        # If user has no test purchases, skip
        if true_products.sum() == 0:
            continue
        
        # Create score array: 1 for predicted items in rank order, 0 otherwise
        pred_scores = np.zeros(n_products)
        for rank, product_idx in enumerate(pred_products):
            # Higher scores for items ranked earlier
            pred_scores[product_idx] = len(pred_products) - rank
        
        # Calculate NDCG for each K
        for k in k_values:
            try:
                ndcg = ndcg_score([true_products], [pred_scores], k=k)
                ndcg_scores[f'ndcg@{k}'].append(ndcg)
            except:
                # Handle edge cases
                pass
    
    # Average across all users
    results = {}
    for k in k_values:
        key = f'ndcg@{k}'
        if len(ndcg_scores[key]) > 0:
            results[key] = np.mean(ndcg_scores[key])
        else:
            results[key] = 0.0
    
    return results


def evaluate_model(predictions_dict, test_matrix, k_values=[5, 10]):
    """
    Comprehensive model evaluation.
    
    Calculates multiple metrics:
    - NDCG@K (primary metric)
    - Precision@K
    - Recall@K
    - Coverage
    
    Parameters:
    -----------
    predictions_dict : dict
        {user_idx: [list of product_idx]}
    test_matrix : scipy sparse matrix
        Ground truth test set
    k_values : list
        K values to evaluate
    
    Returns:
    --------
    dict : All evaluation metrics
    """
    
    print("\nEvaluating model performance...")
    
    results = {}
    
    # NDCG (primary metric)
    ndcg_results = calculate_ndcg(predictions_dict, test_matrix, k_values)
    results.update(ndcg_results)
    
    # Precision and Recall
    for k in k_values:
        precision_scores = []
        recall_scores = []
        
        for user_idx, pred_items in predictions_dict.items():
            # Limit to top K
            pred_k = pred_items[:k]
            
            # Ground truth items
            true_items = test_matrix[user_idx].nonzero()[1]
            
            if len(true_items) == 0:
                continue
            
            # Calculate hits
            hits = len(set(pred_k) & set(true_items))
            
            precision = hits / k if k > 0 else 0
            recall = hits / len(true_items) if len(true_items) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        results[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0
        results[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0
    
    # Coverage: what fraction of items are ever recommended?
    all_recommended_items = set()
    for pred_items in predictions_dict.values():
        all_recommended_items.update(pred_items)
    
    results['coverage'] = len(all_recommended_items) / test_matrix.shape[1]
    
    return results


def compare_to_random_baseline(test_matrix, n_items, k=10, n_samples=1000):
    """
    Calculate baseline NDCG from random recommendations.
    
    This provides context for interpreting model performance.
    A good model should be 5-10x better than random.
    
    Parameters:
    -----------
    test_matrix : scipy sparse matrix
        Ground truth test set
    n_items : int
        Total number of products
    k : int
        Number of recommendations
    n_samples : int
        Number of users to sample for baseline (default 1000 for speed)
    
    Returns:
    --------
    dict : Baseline NDCG scores
    """
    
    print("\nCalculating random baseline...")
    
    # Sample users for efficiency
    n_users = test_matrix.shape[0]
    sample_users = np.random.choice(n_users, min(n_samples, n_users), replace=False)
    
    print(f"  Sampling {len(sample_users):,} users (out of {n_users:,}) for speed")
    
    # Generate random predictions
    random_predictions = {}
    for user_idx in sample_users:
        # Random ranking of all products
        random_predictions[user_idx] = np.random.permutation(n_items)[:k].tolist()
    
    # Evaluate
    baseline_results = calculate_ndcg(random_predictions, test_matrix, k_values=[k])
    
    return baseline_results


# =============================================================================
# COLD START HANDLING
# =============================================================================

def get_popular_items(train_matrix, top_k=10):
    """
    Get most popular products from training data.
    
    Used as fallback recommendations for cold start users.
    
    Parameters:
    -----------
    train_matrix : scipy sparse matrix
        Training interaction matrix
    top_k : int
        Number of popular items to return
    
    Returns:
    --------
    list : Indices of top-k most popular products
    """
    
    # Sum purchases per product
    product_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    
    # Get top K
    popular_indices = np.argsort(product_popularity)[::-1][:top_k]
    
    return popular_indices.tolist()


def handle_cold_start(predictions_dict, test_users, popular_items):
    """
    Add popular items for users without predictions.
    
    For cold start users (new users or users without training data),
    recommend the most popular items as a fallback strategy.
    
    Parameters:
    -----------
    predictions_dict : dict
        Existing predictions {user_idx: [product_idx]}
    test_users : list
        All user indices that need predictions
    popular_items : list
        Popular product indices to use as fallback
    
    Returns:
    --------
    dict : Updated predictions with cold start handling
    """
    
    predictions_with_fallback = predictions_dict.copy()
    
    cold_start_count = 0
    for user_idx in test_users:
        if user_idx not in predictions_with_fallback:
            predictions_with_fallback[user_idx] = popular_items
            cold_start_count += 1
    
    if cold_start_count > 0:
        print(f"  Applied cold start fallback for {cold_start_count} users")
    
    return predictions_with_fallback


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_ndcg_comparison(results_dict, save_path=None):
    """
    Create bar plot comparing NDCG scores across models.
    
    Parameters:
    -----------
    results_dict : dict
        {model_name: {metric: value}}
    save_path : str, optional
        Path to save figure
    """
    
    # Extract NDCG scores
    models = list(results_dict.keys())
    ndcg_5 = [results_dict[m].get('ndcg@5', 0) for m in models]
    ndcg_10 = [results_dict[m].get('ndcg@10', 0) for m in models]
    
    # Create plot
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ndcg_5, width, label='NDCG@5', alpha=0.8)
    bars2 = ax.bar(x + width/2, ndcg_10, width, label='NDCG@10', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('NDCG Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: NDCG Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, max(max(ndcg_5), max(ndcg_10)) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close without showing to prevent blocking
    else:
        plt.show()  # Only show if not saving


def visualize_embeddings(embeddings, product_info, method='pca', n_samples=500, save_path=None):
    """
    Visualize item embeddings in 2D.
    
    Useful for understanding what the model learned about product relationships.
    Similar products (e.g., dairy items) should cluster together.
    
    Parameters:
    -----------
    embeddings : numpy array
        Item embeddings matrix (n_items × embedding_dim)
    product_info : DataFrame
        Product metadata including names and departments
    method : str
        'pca' or 'tsne'
    n_samples : int
        Number of products to plot (for clarity)
    save_path : str, optional
        Path to save figure
    """
    
    print(f"\nVisualizing embeddings using {method.upper()}...")
    
    # Sample products if too many
    n_products = embeddings.shape[0]
    if n_products > n_samples:
        sample_idx = np.random.choice(n_products, n_samples, replace=False)
        embeddings_sample = embeddings[sample_idx]
        products_sample = product_info.iloc[sample_idx]
    else:
        embeddings_sample = embeddings
        products_sample = product_info
    
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    
    embeddings_2d = reducer.fit_transform(embeddings_sample)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by department
    departments = products_sample['department'].unique()
    colors = sns.color_palette('husl', n_colors=len(departments))
    dept_to_color = {dept: colors[i] for i, dept in enumerate(departments)}
    
    for dept in departments:
        mask = products_sample['department'] == dept
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  label=dept, alpha=0.6, s=50)
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title('Product Embedding Visualization\nSimilar products cluster together', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close without showing
    else:
        plt.show()  # Only show if not saving


def show_sample_recommendations(predictions_dict, product_info, 
                                user_features=None, n_samples=5):
    """
    Display recommendations for sample users.
    
    Shows actual product names to make results interpretable.
    
    Parameters:
    -----------
    predictions_dict : dict
        {user_idx: [product_idx list]}
    product_info : DataFrame
        Product metadata
    user_features : DataFrame, optional
        User features for context
    n_samples : int
        Number of users to show
    """
    
    print("\n" + "="*70)
    print("SAMPLE RECOMMENDATIONS")
    print("="*70)
    
    # Sample users
    sample_users = np.random.choice(list(predictions_dict.keys()), 
                                   min(n_samples, len(predictions_dict)), 
                                   replace=False)
    
    for user_idx in sample_users:
        print(f"\nUser {user_idx}:")
        
        # Show user info if available
        if user_features is not None:
            user_info = user_features[user_features['user_idx'] == user_idx]
            if not user_info.empty:
                n_orders = user_info['n_orders'].values[0]
                avg_basket = user_info['avg_basket_size'].values[0]
                print(f"  Profile: {n_orders} orders, avg {avg_basket:.1f} items per order")
        
        # Show recommendations
        pred_indices = predictions_dict[user_idx][:10]
        
        print("  Top 10 Recommendations:")
        for i, prod_idx in enumerate(pred_indices, 1):
            prod_row = product_info[product_info['product_idx'] == prod_idx]
            if not prod_row.empty:
                name = prod_row['product_name'].values[0]
                dept = prod_row['department'].values[0]
                print(f"    {i:2d}. {name} ({dept})")


def print_evaluation_summary(results, model_name="Model"):
    """
    Print formatted evaluation results.
    
    Parameters:
    -----------
    results : dict
        Evaluation metrics
    model_name : str
        Name of the model for display
    """
    
    print("\n" + "="*70)
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print("="*70)
    
    # NDCG scores
    print("\nNDCG Scores (Primary Metric):")
    for k in [5, 10]:
        key = f'ndcg@{k}'
        if key in results:
            print(f"  NDCG@{k:2d} = {results[key]:.4f}")
    
    # Precision and Recall
    print("\nPrecision & Recall:")
    for k in [5, 10]:
        prec_key = f'precision@{k}'
        rec_key = f'recall@{k}'
        if prec_key in results and rec_key in results:
            print(f"  @{k:2d}: Precision = {results[prec_key]:.4f}, Recall = {results[rec_key]:.4f}")
    
    # Coverage
    if 'coverage' in results:
        print(f"\nCatalog Coverage: {results['coverage']:.2%}")
    
    print("="*70)


def save_results(results, model_name, output_path):
    """
    Save evaluation results to disk.
    
    Allows Tutorial 3 to load and compare against Tutorial 2 results.
    
    Parameters:
    -----------
    results : dict
        Evaluation metrics
    model_name : str
        Model identifier
    output_path : str
        Path to save results
    """
    
    results_with_metadata = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': results
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results_with_metadata, f)
    
    print(f"\nResults saved to: {output_path}")


def load_previous_results(results_path):
    """
    Load previously saved results.
    
    Used in Tutorial 3 to compare against Tutorial 2.
    
    Parameters:
    -----------
    results_path : str
        Path to saved results
    
    Returns:
    --------
    dict : Previous results
    """
    
    with open(results_path, 'rb') as f:
        results_data = pickle.load(f)
    
    return results_data['metrics']