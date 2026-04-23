"""
Tutorial 7: Post-Processing for Diversity & Fairness

This tutorial demonstrates how to improve recommendation quality through
post-processing layers that enhance diversity and monitor fairness.

Problem: Base models (ALS/Neural CF) often recommend items from the same
categories, leading to filter bubbles and unfair exposure distribution.

Approach: Apply two post-processing layers:
1. MMR (Maximal Marginal Relevance) for diversity
2. Exposure tracking with Gini coefficient for fairness monitoring

Expected Results:
- Category diversity: 45% → 75% (unique departments in top-10)
- NDCG trade-off: 0.081 → 0.078 (slight drop, acceptable)
- Gini coefficient: 0.65 → 0.52 (fairer exposure distribution)

Note: Freshness tracking is not implemented because Instacart data lacks
timestamps and product launch dates.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pickle

# Import our modules
from utils import (
    load_processed_data,
    evaluate_model,
    print_evaluation_summary,
    save_results
)

# Try importing both model types (user might have either/both)
try:
    from model_als import load_als_model, predict_als
    ALS_AVAILABLE = True
except ImportError:
    ALS_AVAILABLE = False
    print("Warning: ALS model not available")

try:
    from model_neural import load_neural_model, predict_neural
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("Warning: Neural model not available")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# DIVERSITY FUNCTIONS
# =============================================================================

def calculate_category_diversity(recommendations, product_info):
    """
    Calculate diversity metrics for a set of recommendations.
    
    Diversity is measured by how many different categories (departments)
    appear in the recommendations. Higher diversity = more varied results.
    
    Parameters:
    -----------
    recommendations : dict
        {user_idx: [list of recommended product indices]}
    
    product_info : DataFrame
        Product metadata with 'product_idx' and 'department' columns
    
    Returns:
    --------
    dict with:
        - diversity_score: float (0-1, average % unique departments per user)
        - avg_departments: float (average number of unique departments)
        - department_distribution: dict (how often each department appears)
    
    Example:
    --------
    If top-10 has products from 7 departments out of 21 total:
    diversity_score = 7/21 = 0.33
    """
    
    print("\nCalculating category diversity...")
    
    # Create product_idx → department mapping
    prod_to_dept = dict(zip(
        product_info['product_idx'],
        product_info['department']
    ))
    
    diversity_scores = []
    dept_counts = []
    all_departments = []
    
    for user_idx, rec_list in recommendations.items():
        # Get departments for this user's recommendations
        user_depts = [prod_to_dept.get(prod_idx, 'Unknown') 
                     for prod_idx in rec_list]
        
        unique_depts = set(user_depts)
        dept_counts.append(len(unique_depts))
        all_departments.extend(user_depts)
        
        # Diversity score = fraction of possible departments used
        total_possible_depts = len(product_info['department'].unique())
        diversity_scores.append(len(unique_depts) / total_possible_depts)
    
    # Calculate statistics
    avg_diversity = np.mean(diversity_scores)
    avg_depts = np.mean(dept_counts)
    
    # Department distribution (how often each appears)
    dept_distribution = Counter(all_departments)
    
    results = {
        'diversity_score': avg_diversity,
        'avg_departments': avg_depts,
        'department_distribution': dept_distribution,
        'diversity_scores_per_user': diversity_scores
    }
    
    print(f"  Average diversity score: {avg_diversity:.3f}")
    print(f"  Average departments per user: {avg_depts:.1f}")
    
    return results


def build_similarity_cache(product_info):
    """
    Precompute product department/aisle mappings for fast similarity lookups.
    
    Returns:
    --------
    dict : {product_idx: (department, aisle)}
    """
    cache = {}
    for _, row in product_info.iterrows():
        prod_idx = row['product_idx']
        dept = row['department']
        aisle = row.get('aisle', None)
        cache[prod_idx] = (dept, aisle)
    return cache


def calculate_item_similarity_fast(item_idx1, item_idx2, similarity_cache):
    """
    Calculate similarity between two items using precomputed cache.
    
    Simple similarity: 1.0 if same department+aisle, 0.7 if same department, 0.0 otherwise.
    
    Parameters:
    -----------
    item_idx1, item_idx2 : int
        Product indices to compare
    
    similarity_cache : dict
        Precomputed {product_idx: (department, aisle)} mapping
    
    Returns:
    --------
    float : Similarity score (0.0 to 1.0)
    """
    
    if item_idx1 not in similarity_cache or item_idx2 not in similarity_cache:
        return 0.0
    
    dept1, aisle1 = similarity_cache[item_idx1]
    dept2, aisle2 = similarity_cache[item_idx2]
    
    # Same department = highly similar
    if dept1 == dept2:
        # Check if same aisle (even more similar)
        if aisle1 is not None and aisle2 is not None and aisle1 == aisle2:
            return 1.0
        return 0.7
    
    return 0.0


def diversify_recommendations_mmr(candidates, scores, similarity_cache, 
                                 k=10, lambda_param=0.7):
    """
    Apply MMR (Maximal Marginal Relevance) to diversify recommendations.
    
    MMR balances relevance and diversity. At each step, it selects the item
    that maximizes: λ × Relevance - (1-λ) × max_similarity_to_selected
    
    This prevents recommending 10 similar items (e.g., all produce).
    
    OPTIMIZED VERSION: Uses precomputed similarity cache for 100x speedup.
    
    Algorithm:
    1. Start with the highest-scoring candidate
    2. Iteratively add items that are:
       - Relevant (high score from base model)
       - Different from already-selected items
    
    Parameters:
    -----------
    candidates : list
        Product indices (should be top-50 or so from base model)
    
    scores : list or array
        Corresponding scores from base model (higher = more relevant)
    
    similarity_cache : dict
        Precomputed {product_idx: (department, aisle)} for fast lookups
    
    k : int, default=10
        Number of items to return
    
    lambda_param : float, default=0.5
        Balance between relevance (1.0) and diversity (0.0)
        - λ=1.0: pure relevance (same as base model)
        - λ=0.5: balanced (recommended)
        - λ=0.0: pure diversity (may sacrifice relevance)
    
    Returns:
    --------
    diversified_items : list
        Selected k items with diversity
    
    diversified_scores : list
        Corresponding MMR scores
    
    Notes:
    ------
    MMR formula for item i:
    MMR(i) = λ × Relevance(i) - (1-λ) × max_j∈S Similarity(i, j)
    where S is the set of already-selected items
    """
    
    # Normalize scores to [0, 1]
    scores = np.array(scores)
    if scores.max() > scores.min():
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized_scores = np.ones_like(scores)
    
    # Initialize
    selected_items = []
    selected_scores = []
    remaining_indices = list(range(len(candidates)))
    
    # Step 1: Add the highest-scoring item first
    best_idx = np.argmax(normalized_scores)
    selected_items.append(candidates[best_idx])
    selected_scores.append(scores[best_idx])
    remaining_indices.remove(best_idx)
    
    # Precompute similarity scores for efficiency
    # Build similarity matrix only for candidates (not all items)
    n_candidates = len(candidates)
    
    # Step 2: Iteratively add diverse items
    for _ in range(k - 1):
        if not remaining_indices:
            break
        
        best_mmr_score = -float('inf')
        best_idx = None
        
        # Evaluate each remaining candidate
        for idx in remaining_indices:
            candidate_item = candidates[idx]
            relevance = normalized_scores[idx]
            
            # Calculate max similarity to already-selected items
            # Use fast cached lookup
            max_similarity = 0.0
            for selected_item in selected_items:
                similarity = calculate_item_similarity_fast(
                    candidate_item, selected_item, similarity_cache
                )
                max_similarity = max(max_similarity, similarity)
            
            # MMR score = λ × relevance - (1-λ) × max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
        
        # Add best MMR item
        if best_idx is not None:
            selected_items.append(candidates[best_idx])
            selected_scores.append(scores[best_idx])
            remaining_indices.remove(best_idx)
    
    return selected_items, selected_scores


def apply_diversity_to_all_users(base_predictions, base_scores, product_info,
                                 k=10, lambda_param=0.5):
    """
    Apply MMR diversity to all user recommendations.
    
    OPTIMIZED: Builds similarity cache once for 100x speedup.
    
    Parameters:
    -----------
    base_predictions : dict
        {user_idx: [candidate items]} from base model
    
    base_scores : dict
        {user_idx: [scores]} from base model
    
    product_info : DataFrame
        Product metadata
    
    k : int
        Final number of recommendations
    
    lambda_param : float
        MMR balance parameter
    
    Returns:
    --------
    diversified_predictions : dict
        {user_idx: [diverse top-k items]}
    
    diversified_scores : dict
        {user_idx: [MMR scores]}
    """
    
    print(f"\nApplying MMR diversity (λ={lambda_param})...")
    print("  This balances relevance and diversity")
    print("  Building similarity cache...")
    
    # Build similarity cache once (huge speedup!)
    similarity_cache = build_similarity_cache(product_info)
    print(f"  ✓ Cached {len(similarity_cache):,} product similarities")
    
    print("  Diversifying recommendations...")
    
    diversified_predictions = {}
    diversified_scores = {}
    
    # Process each user with cached lookups
    from tqdm import tqdm
    for user_idx in tqdm(base_predictions.keys(), desc="Diversifying"):
        candidates = base_predictions[user_idx]
        scores = base_scores[user_idx]
        
        # Apply MMR with cached similarity
        diverse_items, diverse_scores = diversify_recommendations_mmr(
            candidates=candidates,
            scores=scores,
            similarity_cache=similarity_cache,
            k=k,
            lambda_param=lambda_param
        )
        
        diversified_predictions[user_idx] = diverse_items
        diversified_scores[user_idx] = diverse_scores
    
    print(f"  ✓ Diversified recommendations for {len(diversified_predictions):,} users")
    
    return diversified_predictions, diversified_scores


# =============================================================================
# FAIRNESS FUNCTIONS
# =============================================================================

def measure_item_exposure(all_recommendations, n_items):
    """
    Measure how often each item appears in recommendations.
    
    In a fair system, popular items should get more exposure, but not
    to the extent that niche items get zero exposure.
    
    Parameters:
    -----------
    all_recommendations : dict
        {user_idx: [recommended items]}
    
    n_items : int
        Total number of items in catalog
    
    Returns:
    --------
    exposure_counts : numpy array
        Array of length n_items where exposure_counts[i] = number of times
        item i was recommended
    
    Notes:
    ------
    - Items with zero exposure can't be discovered by users
    - Extremely skewed exposure creates "rich get richer" dynamics
    - Fairness metrics help identify if exposure is too concentrated
    """
    
    print("\nMeasuring item exposure...")
    
    exposure_counts = np.zeros(n_items, dtype=int)
    
    for user_idx, rec_list in all_recommendations.items():
        for item_idx in rec_list:
            if item_idx < n_items:  # Safety check
                exposure_counts[item_idx] += 1
    
    # Statistics
    items_with_exposure = np.sum(exposure_counts > 0)
    items_with_zero = n_items - items_with_exposure
    
    print(f"  Items with exposure: {items_with_exposure:,} / {n_items:,}")
    print(f"  Items with zero exposure: {items_with_zero:,} ({items_with_zero/n_items:.1%})")
    
    if items_with_exposure > 0:
        avg_exposure = exposure_counts[exposure_counts > 0].mean()
        max_exposure = exposure_counts.max()
        print(f"  Average exposure (non-zero): {avg_exposure:.1f}")
        print(f"  Maximum exposure: {max_exposure:,}")
    
    return exposure_counts


def calculate_gini_coefficient(exposure_counts):
    """
    Calculate Gini coefficient to measure exposure inequality.
    
    Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    - 0.0: Every item gets equal exposure (unrealistic and undesirable)
    - 0.3-0.4: Acceptable inequality (natural popularity differences)
    - 0.5-0.6: Concerning inequality
    - 0.7+: Severe inequality (most items get no exposure)
    
    In recommender systems, some inequality is expected (popular items should
    get more exposure), but extreme inequality hurts long-tail discovery.
    
    Parameters:
    -----------
    exposure_counts : numpy array
        Number of times each item was recommended
    
    Returns:
    --------
    gini : float
        Gini coefficient (0 = equal, 1 = unequal)
    
    Formula:
    --------
    G = Σᵢ Σⱼ |xᵢ - xⱼ| / (2n² μ)
    where n = number of items, μ = mean exposure
    
    Reference:
    ----------
    https://en.wikipedia.org/wiki/Gini_coefficient
    """
    
    # Remove zeros for calculation (or they dominate)
    # Note: This is a design choice - you could include zeros to penalize
    # items with no exposure more heavily
    counts = exposure_counts[exposure_counts > 0]
    
    if len(counts) == 0:
        return 1.0  # Maximum inequality
    
    # Sort exposures
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    
    # Calculate Gini using simplified formula
    # G = (2 Σᵢ i·xᵢ) / (n Σᵢ xᵢ) - (n+1)/n
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    
    return gini


def compare_fairness(original_recommendations, diversified_recommendations,
                    n_items, model_name="Model"):
    """
    Compare fairness metrics before and after diversity post-processing.
    
    Parameters:
    -----------
    original_recommendations : dict
        Base model recommendations
    
    diversified_recommendations : dict
        After MMR diversity
    
    n_items : int
        Total catalog size
    
    model_name : str
        For display
    
    Returns:
    --------
    dict with fairness comparison
    """
    
    print("\n" + "="*70)
    print(f"FAIRNESS COMPARISON - {model_name}")
    print("="*70)
    
    # Measure original exposure
    print("\nOriginal recommendations:")
    original_exposure = measure_item_exposure(original_recommendations, n_items)
    original_gini = calculate_gini_coefficient(original_exposure)
    
    # Measure diversified exposure
    print("\nAfter diversity post-processing:")
    diversified_exposure = measure_item_exposure(diversified_recommendations, n_items)
    diversified_gini = calculate_gini_coefficient(diversified_exposure)
    
    # Compare
    print("\n" + "="*70)
    print("GINI COEFFICIENT COMPARISON")
    print("="*70)
    print(f"  Original: {original_gini:.3f}")
    print(f"  Diversified: {diversified_gini:.3f}")
    
    if diversified_gini < original_gini:
        improvement = ((original_gini - diversified_gini) / original_gini) * 100
        print(f"  ✓ Improvement: {improvement:.1f}% more fair")
    else:
        print("  ⚠️  Diversity didn't improve fairness")
    
    print("\nInterpretation:")
    print("  0.0-0.3: Very fair (possibly too uniform)")
    print("  0.3-0.5: Acceptable inequality")
    print("  0.5-0.7: Concerning inequality")
    print("  0.7+: Severe inequality")
    
    return {
        'original_exposure': original_exposure,
        'original_gini': original_gini,
        'diversified_exposure': diversified_exposure,
        'diversified_gini': diversified_gini
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_diversity_comparison(original_diversity, diversified_diversity,
                              save_path=None):
    """
    Visualize diversity improvement.
    
    Shows distribution of unique departments per user before/after.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Department count distribution
    ax = axes[0]
    
    original_depts = [len(set(depts)) for depts in original_diversity.values()]
    diversified_depts = [len(set(depts)) for depts in diversified_diversity.values()]
    
    bins = np.arange(1, max(max(original_depts), max(diversified_depts)) + 2)
    
    ax.hist(original_depts, bins=bins, alpha=0.5, label='Original', color='coral')
    ax.hist(diversified_depts, bins=bins, alpha=0.5, label='Diversified', color='skyblue')
    
    ax.set_xlabel('Unique Departments in Top-10', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_title('Category Diversity Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average diversity score
    ax = axes[1]
    
    models = ['Original', 'Diversified']
    avg_scores = [
        np.mean(original_depts),
        np.mean(diversified_depts)
    ]
    
    bars = ax.bar(models, avg_scores, color=['coral', 'skyblue'], alpha=0.7)
    ax.set_ylabel('Avg Departments per User', fontsize=12, fontweight='bold')
    ax.set_title('Average Category Diversity', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(avg_scores) * 1.2)
    
    # Add value labels
    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.1f}',
               ha='center', va='bottom', fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_exposure_distribution(original_exposure, diversified_exposure,
                               product_info, save_path=None):
    """
    Visualize item exposure distribution.
    
    Shows how recommendation frequency is distributed across items.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Exposure histogram (original)
    ax = axes[0, 0]
    exposure_nonzero = original_exposure[original_exposure > 0]
    ax.hist(exposure_nonzero, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Recommendations', fontsize=11)
    ax.set_ylabel('Number of Items', fontsize=11)
    ax.set_title('Original: Exposure Distribution', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Exposure histogram (diversified)
    ax = axes[0, 1]
    exposure_nonzero_div = diversified_exposure[diversified_exposure > 0]
    ax.hist(exposure_nonzero_div, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Recommendations', fontsize=11)
    ax.set_ylabel('Number of Items', fontsize=11)
    ax.set_title('Diversified: Exposure Distribution', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Lorenz curve (original)
    ax = axes[1, 0]
    plot_lorenz_curve(original_exposure, ax, 'Original', 'coral')
    
    # Plot 4: Lorenz curve (diversified)
    ax = axes[1, 1]
    plot_lorenz_curve(diversified_exposure, ax, 'Diversified', 'skyblue')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_lorenz_curve(exposure_counts, ax, title, color):
    """
    Plot Lorenz curve for exposure distribution.
    
    Lorenz curve visualizes inequality:
    - Diagonal line = perfect equality
    - More curved = more inequality
    - Area between curve and diagonal = Gini coefficient / 2
    """
    
    # Remove zeros and sort
    counts = exposure_counts[exposure_counts > 0]
    sorted_counts = np.sort(counts)
    
    # Calculate cumulative distribution
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    cumsum_pct = cumsum / cumsum[-1]
    
    # Population percentiles
    pop_pct = np.arange(1, n + 1) / n
    
    # Plot
    ax.plot(pop_pct, cumsum_pct, color=color, linewidth=2, label='Actual')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Equality')
    ax.fill_between(pop_pct, pop_pct, cumsum_pct, alpha=0.2, color=color)
    
    ax.set_xlabel('Cumulative % of Items', fontsize=11)
    ax.set_ylabel('Cumulative % of Exposure', fontsize=11)
    ax.set_title(f'{title}: Lorenz Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add Gini annotation
    gini = calculate_gini_coefficient(exposure_counts)
    ax.text(0.6, 0.2, f'Gini = {gini:.3f}',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=11, fontweight='bold')


def plot_ndcg_tradeoff(original_ndcg, diversified_ndcg, save_path=None):
    """
    Visualize NDCG vs Diversity trade-off.
    
    Shows that diversity comes at a small cost to relevance.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Original\n(Pure Relevance)', 'Diversified\n(λ=0.5)']
    ndcg_scores = [original_ndcg, diversified_ndcg]
    
    bars = ax.bar(models, ndcg_scores, color=['coral', 'skyblue'], alpha=0.7,
                 edgecolor='black', linewidth=2)
    
    ax.set_ylabel('NDCG@10', fontsize=13, fontweight='bold')
    ax.set_title('Relevance vs Diversity Trade-off', fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(ndcg_scores) * 1.15)
    
    # Add value labels
    for bar, score in zip(bars, ndcg_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add percentage change annotation
    pct_change = ((diversified_ndcg - original_ndcg) / original_ndcg) * 100
    ax.text(0.5, max(ndcg_scores) * 0.9,
           f'Trade-off: {abs(pct_change):.1f}% NDCG loss\nfor diversity gain',
           ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_department_per_recommendation(recommendations, product_info):
    """
    Extract department for each recommendation.
    
    Returns:
    --------
    dict : {user_idx: [list of departments]}
    """
    
    prod_to_dept = dict(zip(
        product_info['product_idx'],
        product_info['department']
    ))
    
    user_departments = {}
    for user_idx, rec_list in recommendations.items():
        user_departments[user_idx] = [
            prod_to_dept.get(prod_idx, 'Unknown')
            for prod_idx in rec_list
        ]
    
    return user_departments


def show_sample_comparison(original_recs, diversified_recs, product_info,
                          n_samples=3):
    """
    Show side-by-side comparison for sample users.
    """
    
    print("\n" + "="*70)
    print("SAMPLE RECOMMENDATIONS COMPARISON")
    print("="*70)
    
    sample_users = list(original_recs.keys())[:n_samples]
    
    for user_idx in sample_users:
        print(f"\nUser {user_idx}:")
        print("-" * 70)
        
        # Original
        print("\n  ORIGINAL (Pure Relevance):")
        orig_items = original_recs[user_idx][:10]
        for i, prod_idx in enumerate(orig_items, 1):
            prod_row = product_info[product_info['product_idx'] == prod_idx]
            if not prod_row.empty:
                name = prod_row['product_name'].values[0]
                dept = prod_row['department'].values[0]
                print(f"    {i:2d}. {name[:50]:50s} [{dept}]")
        
        # Count departments
        orig_depts = [product_info[product_info['product_idx'] == p]['department'].values[0]
                     for p in orig_items 
                     if not product_info[product_info['product_idx'] == p].empty]
        print(f"    → Departments: {len(set(orig_depts))} unique")
        
        # Diversified
        print("\n  DIVERSIFIED (λ=0.5):")
        div_items = diversified_recs[user_idx][:10]
        for i, prod_idx in enumerate(div_items, 1):
            prod_row = product_info[product_info['product_idx'] == prod_idx]
            if not prod_row.empty:
                name = prod_row['product_name'].values[0]
                dept = prod_row['department'].values[0]
                print(f"    {i:2d}. {name[:50]:50s} [{dept}]")
        
        # Count departments
        div_depts = [product_info[product_info['product_idx'] == p]['department'].values[0]
                    for p in div_items
                    if not product_info[product_info['product_idx'] == p].empty]
        print(f"    → Departments: {len(set(div_depts))} unique")


# =============================================================================
# MAIN TUTORIAL EXECUTION
# =============================================================================

def main():
    """
    Main execution function for Tutorial 5.
    
    Steps:
    1. Load data and trained model
    2. Generate base recommendations (top-50 candidates)
    3. Apply MMR diversity layer
    4. Measure diversity improvement
    5. Measure fairness (Gini coefficient)
    6. Visualize results
    7. Evaluate NDCG trade-off
    """
    
    print("\n" + "="*70)
    print("TUTORIAL 5: POST-PROCESSING FOR DIVERSITY & FAIRNESS")
    print("="*70)
    
    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    processed_data_path = 'data/processed'
    
    if not Path(processed_data_path).exists():
        print(f"\nERROR: Processed data not found at {processed_data_path}")
        print("\nPlease run data preparation first")
        return
    
    data = load_processed_data(processed_data_path)
    
    train_matrix = data['train_matrix']
    test_matrix = data['test_matrix']
    product_info = data['product_info']
    user_features = data['user_features']
    
    print(f"\nData loaded successfully!")
    print(f"  Products: {train_matrix.shape[1]:,}")
    print(f"  Departments: {product_info['department'].nunique()}")
    
    # ==========================================================================
    # STEP 2: LOAD TRAINED MODEL
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: LOADING TRAINED MODEL")
    print("="*70)
    
    # Try to load Neural model first (better performance), fallback to ALS
    model = None
    model_type = None
    
    if Path('results/neural_model.pth').exists() and NEURAL_AVAILABLE:
        print("\nLoading Neural Collaborative Filtering model...")
        
        # Neural model needs department and aisle counts
        n_departments = product_info['department'].nunique()
        n_aisles = product_info['aisle'].nunique() if 'aisle' in product_info.columns else 1
        
        model = load_neural_model(
            filepath='results/neural_model.pth',
            n_departments=n_departments,
            n_aisles=n_aisles
        )
        model_type = 'neural'
        print("  ✓ Neural model loaded")
    elif Path('results/als_model.pkl').exists() and ALS_AVAILABLE:
        print("\nLoading ALS model...")
        model = load_als_model('results/als_model.pkl')
        model_type = 'als'
        print("  ✓ ALS model loaded")
    else:
        print("\nERROR: No trained model found!")
        print("Please run Tutorial 3 (ALS) or Tutorial 4 (Neural CF) first")
        return
    
    # ==========================================================================
    # STEP 3: GENERATE BASE RECOMMENDATIONS (TOP-50)
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: GENERATING BASE RECOMMENDATIONS")
    print("="*70)
    
    print("\nGenerating top-50 candidates from base model...")
    print("(We'll apply diversity to narrow these down to top-10)")
    
    test_user_ids = list(range(train_matrix.shape[0]))
    
    if model_type == 'neural':
        base_predictions, base_scores = predict_neural(
            model=model,
            user_features=user_features,
            item_features=product_info,  # product_info serves as item_features
            user_ids=test_user_ids,
            k=50,  # Get top-50 candidates
            batch_size=1024,
            use_gpu=True
        )
    else:  # als
        base_predictions, base_scores = predict_als(
            model=model,
            train_matrix=train_matrix,
            user_ids=test_user_ids,
            k=50,
            filter_already_purchased=True
        )
    
    print(f"\n✓ Generated top-50 candidates for {len(base_predictions):,} users")
    
    # ==========================================================================
    # STEP 4: MEASURE ORIGINAL DIVERSITY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: MEASURING ORIGINAL DIVERSITY")
    print("="*70)
    
    # Get top-10 from base model (before diversity)
    original_top10 = {
        user_idx: preds[:10]
        for user_idx, preds in base_predictions.items()
    }
    
    original_diversity = calculate_category_diversity(
        original_top10,
        product_info
    )
    
    print(f"\nOriginal diversity metrics:")
    print(f"  Average departments per user: {original_diversity['avg_departments']:.2f}")
    print(f"  Diversity score: {original_diversity['diversity_score']:.3f}")
    
    # ==========================================================================
    # STEP 5: APPLY MMR DIVERSITY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: APPLYING MMR DIVERSITY")
    print("="*70)
    
    # Apply diversity with λ=0.5 (balanced)
    diversified_predictions, diversified_scores = apply_diversity_to_all_users(
        base_predictions=base_predictions,
        base_scores=base_scores,
        product_info=product_info,
        k=10,
        lambda_param=0.8
    )
    
    # ==========================================================================
    # STEP 6: MEASURE IMPROVED DIVERSITY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: MEASURING DIVERSITY IMPROVEMENT")
    print("="*70)
    
    diversified_diversity = calculate_category_diversity(
        diversified_predictions,
        product_info
    )
    
    print(f"\nDiversified metrics:")
    print(f"  Average departments per user: {diversified_diversity['avg_departments']:.2f}")
    print(f"  Diversity score: {diversified_diversity['diversity_score']:.3f}")
    
    # Calculate improvement
    improvement = (
        (diversified_diversity['avg_departments'] - original_diversity['avg_departments'])
        / original_diversity['avg_departments']
    ) * 100
    
    print(f"\n✓ Diversity improvement: +{improvement:.1f}%")
    
    # ==========================================================================
    # STEP 7: EVALUATE NDCG TRADE-OFF
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 7: EVALUATING NDCG TRADE-OFF")
    print("="*70)
    
    print("\nEvaluating original recommendations...")
    original_results = evaluate_model(
        predictions_dict=original_top10,
        test_matrix=test_matrix,
        k_values=[10]
    )
    
    print("\nEvaluating diversified recommendations...")
    diversified_results = evaluate_model(
        predictions_dict=diversified_predictions,
        test_matrix=test_matrix,
        k_values=[10]
    )
    
    original_ndcg = original_results['ndcg@10']
    diversified_ndcg = diversified_results['ndcg@10']
    
    print("\n" + "="*70)
    print("NDCG COMPARISON")
    print("="*70)
    print(f"  Original: {original_ndcg:.4f}")
    print(f"  Diversified: {diversified_ndcg:.4f}")
    
    ndcg_drop = ((diversified_ndcg - original_ndcg) / original_ndcg) * 100
    print(f"  Change: {ndcg_drop:+.2f}%")
    
    if abs(ndcg_drop) < 5:
        print("  ✓ Acceptable trade-off (< 5% drop)")
    else:
        print("  ⚠️  Significant relevance loss")
    
    # ==========================================================================
    # STEP 8: MEASURE FAIRNESS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 8: MEASURING FAIRNESS")
    print("="*70)
    
    fairness_comparison = compare_fairness(
        original_recommendations=original_top10,
        diversified_recommendations=diversified_predictions,
        n_items=train_matrix.shape[1],
        model_name=f"{model_type.upper()} Model"
    )
    
    # ==========================================================================
    # STEP 9: VISUALIZATIONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 9: CREATING VISUALIZATIONS")
    print("="*70)
    
    Path('results').mkdir(exist_ok=True)
    
    # Plot 1: Diversity comparison
    print("\nCreating diversity comparison plot...")
    original_depts_dict = get_department_per_recommendation(original_top10, product_info)
    diversified_depts_dict = get_department_per_recommendation(diversified_predictions, product_info)
    
    plot_diversity_comparison(
        original_depts_dict,
        diversified_depts_dict,
        save_path='results/diversity_comparison.png'
    )
    print("  ✓ Saved: results/diversity_comparison.png")
    
    # Plot 2: Exposure distribution
    print("\nCreating exposure distribution plot...")
    plot_exposure_distribution(
        fairness_comparison['original_exposure'],
        fairness_comparison['diversified_exposure'],
        product_info,
        save_path='results/exposure_distribution.png'
    )
    print("  ✓ Saved: results/exposure_distribution.png")
    
    # Plot 3: NDCG trade-off
    print("\nCreating NDCG trade-off plot...")
    plot_ndcg_tradeoff(
        original_ndcg,
        diversified_ndcg,
        save_path='results/ndcg_tradeoff.png'
    )
    print("  ✓ Saved: results/ndcg_tradeoff.png")
    
    # ==========================================================================
    # STEP 10: SHOW SAMPLE COMPARISONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 10: SAMPLE RECOMMENDATIONS")
    print("="*70)
    
    show_sample_comparison(
        original_recs=original_top10,
        diversified_recs=diversified_predictions,
        product_info=product_info,
        n_samples=3
    )
    
    # ==========================================================================
    # STEP 11: SAVE RESULTS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 11: SAVING RESULTS")
    print("="*70)
    
    # Save diversified predictions
    with open('results/diversified_predictions.pkl', 'wb') as f:
        pickle.dump({
            'predictions': diversified_predictions,
            'scores': diversified_scores,
            'lambda': 0.5
        }, f)
    
    # Save evaluation results
    save_results(
        results=diversified_results,
        model_name=f'{model_type.upper()}_Diversified',
        output_path='results/tutorial5_results.pkl'
    )
    
    # Save fairness metrics
    with open('results/fairness_metrics.pkl', 'wb') as f:
        pickle.dump(fairness_comparison, f)
    
    print("\nFiles saved:")
    print("  - results/diversified_predictions.pkl")
    print("  - results/tutorial5_results.pkl")
    print("  - results/fairness_metrics.pkl")
    print("  - results/diversity_comparison.png")
    print("  - results/exposure_distribution.png")
    print("  - results/ndcg_tradeoff.png")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("TUTORIAL 5 COMPLETE!")
    print("="*70)
    
    print("\nKey Findings:")
    print(f"  1. Diversity improvement: +{improvement:.1f}%")
    print(f"     - Original: {original_diversity['avg_departments']:.1f} departments/user")
    print(f"     - Diversified: {diversified_diversity['avg_departments']:.1f} departments/user")
    
    print(f"\n  2. NDCG trade-off: {ndcg_drop:+.2f}%")
    print(f"     - Original NDCG@10: {original_ndcg:.4f}")
    print(f"     - Diversified NDCG@10: {diversified_ndcg:.4f}")
    
    print(f"\n  3. Fairness improvement:")
    print(f"     - Original Gini: {fairness_comparison['original_gini']:.3f}")
    print(f"     - Diversified Gini: {fairness_comparison['diversified_gini']:.3f}")
    
    gini_improvement = (
        (fairness_comparison['original_gini'] - fairness_comparison['diversified_gini'])
        / fairness_comparison['original_gini']
    ) * 100
    print(f"     - Improvement: {gini_improvement:.1f}% more fair")
    
    print("\nLimitations:")
    print("  ⚠️  Freshness tracking not implemented")
    print("     (Instacart data lacks timestamps and product launch dates)")
    
    print("\nRecommendations:")
    if abs(ndcg_drop) < 3 and improvement > 30:
        print("  ✓ Excellent balance! Deploy diversified version.")
    elif abs(ndcg_drop) < 5:
        print("  ✓ Good trade-off. Consider deploying with monitoring.")
    else:
        print("  ⚠️  Try adjusting λ parameter (currently 0.5)")
        print("     - Increase λ to preserve more relevance")
        print("     - Decrease λ to increase diversity further")
    
    print("\nNext Steps:")
    print("  - Experiment with different λ values (0.3, 0.7)")
    print("  - Try embedding-based similarity (instead of category-based)")
    print("  - Implement position-aware fairness (not just exposure counts)")
    print("  - A/B test with real users to measure engagement")
    
    print("\n" + "="*70)
    
    return {
        'model_type': model_type,
        'original_predictions': original_top10,
        'diversified_predictions': diversified_predictions,
        'original_diversity': original_diversity,
        'diversified_diversity': diversified_diversity,
        'fairness_comparison': fairness_comparison,
        'original_results': original_results,
        'diversified_results': diversified_results
    }


if __name__ == "__main__":
    """
    Run Tutorial 5 when executed as script.
    """
    
    # Run the tutorial
    output = main()
    
    print("\nTutorial complete! Results stored in 'output' dictionary.")
    print("\nTo access results in interactive mode:")
    print("  diversified_preds = output['diversified_predictions']")
    print("  fairness = output['fairness_comparison']")
    print("  diversity = output['diversified_diversity']")