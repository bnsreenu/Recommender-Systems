"""
Tutorial 2: Instacart Recommendations with Matrix Factorization (ALS)

This tutorial demonstrates classical collaborative filtering using Alternating
Least Squares (ALS) matrix factorization on the Instacart dataset.

Problem: Given a user's purchase history, recommend the next 10 products
they're likely to buy from our catalog of 800 products.

Approach: ALS learns latent factors for users and items by decomposing the
user-item interaction matrix. Users with similar purchase histories get similar
embeddings, enabling personalized recommendations.

Evaluation: NDCG (from Tutorial 1) measures how well we rank relevant items
at the top of recommendations.

Expected Results:
- Random baseline: NDCG@10 ≈ 0.08
- ALS model: NDCG@10 ≈ 0.25-0.35 (3-4x better than random!)
"""

import numpy as np
import sys
from pathlib import Path

# Import our modules
from utils import (
    load_processed_data,
    evaluate_model,
    compare_to_random_baseline,
    get_popular_items,
    handle_cold_start,
    print_evaluation_summary,
    plot_ndcg_comparison,
    visualize_embeddings,
    show_sample_recommendations,
    save_results
)

from model_als import (
    train_als,
    predict_als,
    get_item_embeddings,
    get_similar_items,
    explain_recommendations,
    get_model_statistics
)


def main():
    """
    Main execution function for Tutorial 2.
    
    Steps:
    1. Load preprocessed data
    2. Train ALS model
    3. Generate recommendations
    4. Evaluate with NDCG
    5. Compare to random baseline
    6. Handle cold start
    7. Visualize results
    8. Save model and results
    """
    
    print("\n" + "="*70)
    print("TUTORIAL 4: COLLABORATIVE FILTERING WITH ALS")
    print("="*70)
    
    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    # Path to processed data (created by utils.load_and_prepare_data)
    processed_data_path = 'data/processed'
    
    # Check if data exists
    if not Path(processed_data_path).exists():
        print(f"\nERROR: Processed data not found at {processed_data_path}")
        print("\nFirst run data preparation:")
        print("  from utils import load_and_prepare_data")
        print("  load_and_prepare_data('data/raw', 'data/processed')")
        return
    
    # Load data
    data = load_processed_data(processed_data_path)
    
    train_matrix = data['train_matrix']
    test_matrix = data['test_matrix']
    mappings = data['mappings']
    product_info = data['product_info']
    user_features = data['user_features']
    
    print(f"\nData loaded successfully!")
    print(f"  Training: {train_matrix.nnz:,} interactions")
    print(f"  Testing: {test_matrix.nnz:,} interactions")
    
    # ==========================================================================
    # STEP 2: TRAIN ALS MODEL
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: TRAINING ALS MODEL")
    print("="*70)
    
    # Train model
    # factors=64: embedding dimension
    # regularization=0.01: prevent overfitting
    # iterations=15: usually converges in 10-20 iterations
    # use_gpu=True: faster training if GPU available
    
    als_model = train_als(
        train_matrix=train_matrix,
        factors=64,
        regularization=0.01,
        iterations=15,
        use_gpu=False,
        alpha=40
    )
    
    # Show model statistics
    print("\nModel Statistics:")
    stats = get_model_statistics(als_model, train_matrix)
    print(f"  Users: {stats['n_users']:,}")
    print(f"  Items: {stats['n_items']:,}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Avg items per user: {stats['avg_items_per_user']:.1f}")
    
    # ==========================================================================
    # STEP 3: GENERATE RECOMMENDATIONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: GENERATING RECOMMENDATIONS")
    print("="*70)
    
    # Generate recommendations for all test users
    test_user_ids = list(range(train_matrix.shape[0]))
    
    predictions, scores = predict_als(
        model=als_model,
        train_matrix=train_matrix,
        user_ids=test_user_ids,
        k=10,
        filter_already_purchased=True
    )
    
    print(f"\nGenerated recommendations for {len(predictions):,} users")
    
    # ==========================================================================
    # STEP 4: EVALUATE MODEL
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: EVALUATING MODEL PERFORMANCE")
    print("="*70)
    
    # Evaluate using NDCG and other metrics
    results = evaluate_model(
        predictions_dict=predictions,
        test_matrix=test_matrix,
        k_values=[5, 10]
    )
    
    # Print formatted results
    print_evaluation_summary(results, model_name="ALS")
    
    # ==========================================================================
    # STEP 5: COMPARE TO RANDOM BASELINE
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: BASELINE COMPARISON")
    print("="*70)
    
    # Calculate random baseline
    random_results = compare_to_random_baseline(
        test_matrix=test_matrix,
        n_items=train_matrix.shape[1],
        k=10,
        n_samples=1000
    )
    
    print("\nRandom Baseline Results:")
    print(f"  NDCG@10 = {random_results['ndcg@10']:.4f}")
    
    print("\nImprovement over Random:")
    improvement = results['ndcg@10'] / random_results['ndcg@10']
    print(f"  ALS is {improvement:.2f}x better than random!")
    
    # Visualize comparison
    Path('results').mkdir(exist_ok=True)
    comparison_data = {
        'Random Baseline': random_results,
        'ALS (Tutorial 2)': results
    }
    plot_ndcg_comparison(comparison_data, save_path='results/als_vs_random.png')
    print("\n✓ Comparison plot saved: results/als_vs_random.png")
    
    # ==========================================================================
    # STEP 6: COLD START HANDLING
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: COLD START HANDLING")
    print("="*70)
    
    # Get popular items for cold start users
    popular_items = get_popular_items(train_matrix, top_k=10)
    
    print("\nMost Popular Products (used for cold start):")
    for i, item_idx in enumerate(popular_items, 1):
        prod_row = product_info[product_info['product_idx'] == item_idx]
        if not prod_row.empty:
            name = prod_row['product_name'].values[0]
            dept = prod_row['department'].values[0]
            print(f"  {i:2d}. {name} ({dept})")
    
    # Apply cold start handling
    predictions_with_fallback = handle_cold_start(
        predictions_dict=predictions,
        test_users=test_user_ids,
        popular_items=popular_items
    )
    
    print("\nCold start strategy: For users without training data,")
    print("recommend the most popular products as a fallback.")
    
    # ==========================================================================
    # STEP 7: SHOW SAMPLE RECOMMENDATIONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 7: SAMPLE RECOMMENDATIONS")
    print("="*70)
    
    # Show recommendations for a few users
    show_sample_recommendations(
        predictions_dict=predictions,
        product_info=product_info,
        user_features=user_features,
        n_samples=5
    )
    
    # ==========================================================================
    # STEP 8: EXPLAIN RECOMMENDATIONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 8: RECOMMENDATION EXPLANATIONS")
    print("="*70)
    
    # Pick a sample user and explain their recommendations
    sample_user = 100  # arbitrary user
    if sample_user in predictions:
        print(f"\nExplaining recommendations for User {sample_user}:")
        
        # Show their purchase history
        user_history = train_matrix[sample_user].nonzero()[1]
        print(f"\nUser's purchase history ({len(user_history)} items):")
        for i, item_idx in enumerate(user_history[:5], 1):  # Show first 5
            prod_row = product_info[product_info['product_idx'] == item_idx]
            if not prod_row.empty:
                name = prod_row['product_name'].values[0]
                print(f"  - {name}")
        if len(user_history) > 5:
            print(f"  ... and {len(user_history) - 5} more")
        
        # Explain top recommendation
        top_rec = predictions[sample_user][0]
        prod_row = product_info[product_info['product_idx'] == top_rec]
        if not prod_row.empty:
            rec_name = prod_row['product_name'].values[0]
            print(f"\nTop recommendation: {rec_name}")
            print("\nWhy this was recommended (based on purchase history):")
            
            explanations = explain_recommendations(
                model=als_model,
                user_idx=sample_user,
                item_idx=top_rec,
                train_matrix=train_matrix,
                product_info=product_info,
                top_n=5
            )
            
            for product_name, score in explanations:
                print(f"  - {product_name} (contribution: {score:.3f})")
    
    # ==========================================================================
    # STEP 9: ITEM SIMILARITY ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 9: ITEM SIMILARITY (\"Customers Also Bought\")")
    print("="*70)
    
    # Pick some popular products and find similar items
    sample_products = popular_items[:3]
    
    similar_items_dict = get_similar_items(
        model=als_model,
        item_ids=sample_products,
        n=5
    )
    
    print("\nProducts frequently bought together:")
    for item_idx, similar_list in similar_items_dict.items():
        prod_row = product_info[product_info['product_idx'] == item_idx]
        if not prod_row.empty:
            base_name = prod_row['product_name'].values[0]
            print(f"\n{base_name}:")
            
            for similar_idx, similarity_score in similar_list:
                sim_prod = product_info[product_info['product_idx'] == similar_idx]
                if not sim_prod.empty:
                    sim_name = sim_prod['product_name'].values[0]
                    print(f"  - {sim_name} (similarity: {similarity_score:.3f})")
    
    # ==========================================================================
    # STEP 10: VISUALIZE EMBEDDINGS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 10: EMBEDDING VISUALIZATION")
    print("="*70)
    
    # Extract item embeddings
    item_embeddings = get_item_embeddings(als_model)
    
    print("\nVisualizing product embeddings...")
    print("Similar products should cluster together (e.g., dairy items, produce items)")
    
    # Visualize using PCA (faster) or t-SNE (better separation)
    visualize_embeddings(
        embeddings=item_embeddings,
        product_info=product_info,
        method='pca',  # or 'tsne' for better visualization (slower)
        n_samples=500,
        save_path='results/als_embeddings.png'
    )
    print("\n✓ Embeddings plot saved: results/als_embeddings.png")
    
    # ==========================================================================
    # STEP 11: SAVE RESULTS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 11: SAVING RESULTS")
    print("="*70)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Save evaluation results for comparison in Tutorial 5
    save_results(
        results=results,
        model_name='ALS',
        output_path='results/als_results.pkl'
    )
    
    # Save model
    from model_als import save_als_model
    save_als_model(als_model, 'results/als_model.pkl')
    
    print("\nFiles saved:")
    print("  - results/als_results.pkl (evaluation metrics)")
    print("  - results/als_model.pkl (trained model)")
    print("  - results/als_vs_random.png (comparison plot)")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("TUTORIAL 2 COMPLETE!")
    print("="*70)
    
    print("\nKey Findings:")
    print(f"  1. ALS NDCG@10: {results['ndcg@10']:.4f}")
    print(f"  2. Random baseline: {random_results['ndcg@10']:.4f}")
    print(f"  3. Improvement: {improvement:.2f}x better than random")
    print(f"  4. Training time: ~1-2 minutes on GPU")
    print(f"  5. Model learns interpretable embeddings")
    
    print("\nInterpretation:")
    if results['ndcg@10'] > 0.30:
        print("  ✅ Excellent performance! Model is production-ready.")
    elif results['ndcg@10'] > 0.20:
        print("  ✅ Good performance. Model shows strong signal.")
    else:
        print("  ⚠️  Moderate performance. Consider feature engineering.")
    
    print("\nNext Steps:")
    print("  - Try different hyperparameters (factors, regularization)")
    print("  - Experiment with BM25 weighting parameters")
    print("  - Compare to Tutorial 3 (Neural CF) to see if deep learning helps")
    
    print("\n" + "="*70)
    
    return {
        'model': als_model,
        'results': results,
        'predictions': predictions
    }


if __name__ == "__main__":
    """
    Run Tutorial 2 when executed as script.
    """
    
    # Run the tutorial
    output = main()
    
    print("\nTutorial complete! Results stored in 'output' dictionary.")
    print("\nTo access results in interactive mode:")
    print("  model = output['model']")
    print("  results = output['results']")
    print("  predictions = output['predictions']")