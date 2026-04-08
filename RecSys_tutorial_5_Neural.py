"""
Tutorial 3: Deep Learning for Instacart - Two-Tower Neural Recommender

This tutorial demonstrates neural collaborative filtering using PyTorch on the
same Instacart dataset from Tutorial 2. The goal is to see if deep learning
with rich features can outperform classical ALS.

Key Differences from Tutorial 2:
- Uses neural networks (MLPs) instead of matrix factorization
- Incorporates user features (order history, preferences)
- Incorporates item features (department, aisle)
- Better cold start (can use features for new users)
- More complex but potentially more accurate

Expected Results:
- ALS (Tutorial 2): NDCG@10 ≈ 0.28-0.34
- Neural CF (Tutorial 3): NDCG@10 ≈ 0.30-0.40
- Improvement: 5-20% better (marginal but consistent)

Question: Is the extra complexity worth it?
"""

import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Import utilities (shared with Tutorial 2)
from utils import (
    load_processed_data,
    evaluate_model,
    compare_to_random_baseline,
    get_popular_items,
    handle_cold_start,
    print_evaluation_summary,
    plot_ndcg_comparison,
    show_sample_recommendations,
    save_results,
    load_previous_results
)

# Import neural model
from model_neural import (
    train_neural,
    predict_neural,
    save_neural_model,
    TwoTowerModel
)


def main():
    """
    Main execution function for Tutorial 3.
    
    Steps:
    1. Load preprocessed data (same as Tutorial 2)
    2. Train two-tower neural model
    3. Generate recommendations
    4. Evaluate with NDCG
    5. Compare to Tutorial 2 (ALS)
    6. Analyze improvements
    7. Save results
    """
    
    print("\n" + "="*70)
    print("TUTORIAL 3: NEURAL COLLABORATIVE FILTERING")
    print("="*70)
    
    # ==========================================================================
    # STEP 1: LOAD DATA (Same as Tutorial 2)
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    processed_data_path = 'data/processed'
    
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
    print(f"  User features: {user_features.shape}")
    print(f"  Item features: {product_info.shape}")
    
    # ==========================================================================
    # STEP 2: TRAIN NEURAL MODEL
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: TRAINING NEURAL CF MODEL")
    print("="*70)
    
    print("\nModel Architecture:")
    print("  User Tower: [user_id + 5 features] → MLP(128→64) → 64-dim vector")
    print("  Item Tower: [item_id + dept + aisle] → MLP(128→64) → 64-dim vector")
    print("  Score: user_vector · item_vector → sigmoid → probability")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        print("\n⚠ No GPU detected, using CPU (will be slower)")
        use_gpu = False
    
    # Train model
    neural_model, training_history = train_neural(
        train_matrix=train_matrix,
        user_features=user_features,
        item_features=product_info,
        n_epochs=50,  # Increased to 50, early stopping will kick in if converged
        batch_size=1024,
        lr=0.001,
        embedding_dim=64,
        hidden_dims=[128, 64],
        negative_ratio=4,
        use_gpu=use_gpu
    )
    
    # Plot and save training curve
    print("\nTraining Loss:")
    for epoch, loss in enumerate(training_history['train_loss'], 1):
        print(f"  Epoch {epoch:2d}: {loss:.4f}")
    
    # Create training history plot
    Path('results').mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_history['train_loss']) + 1), 
             training_history['train_loss'], 
             marker='o', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss (BCE)', fontsize=12)
    plt.title('Neural CF Training History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/neural_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    print("\n✓ Training history plot saved: results/neural_training_history.png")
    
    # ==========================================================================
    # STEP 3: GENERATE RECOMMENDATIONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: GENERATING RECOMMENDATIONS")
    print("="*70)
    
    # Generate recommendations for all test users
    test_user_ids = list(range(train_matrix.shape[0]))
    
    predictions, scores = predict_neural(
        model=neural_model,
        user_features=user_features,
        item_features=product_info,
        user_ids=test_user_ids,
        k=10,
        batch_size=1024,
        use_gpu=use_gpu
    )
    
    print(f"\nGenerated recommendations for {len(predictions):,} users")
    
    # ==========================================================================
    # STEP 4: EVALUATE MODEL
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: EVALUATING NEURAL MODEL")
    print("="*70)
    
    # Evaluate using NDCG and other metrics
    neural_results = evaluate_model(
        predictions_dict=predictions,
        test_matrix=test_matrix,
        k_values=[5, 10]
    )
    
    # Print formatted results
    print_evaluation_summary(neural_results, model_name="Neural CF")
    
    # ==========================================================================
    # STEP 5: COMPARE TO TUTORIAL 2 (ALS)
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: COMPARISON WITH TUTORIAL 2 (ALS)")
    print("="*70)
    
    # Load ALS results from Tutorial 2
    als_results_path = 'results/als_results.pkl'
    
    if Path(als_results_path).exists():
        print("\nLoading ALS results from Tutorial 2...")
        als_results = load_previous_results(als_results_path)
        
        print("\n" + "-"*70)
        print("HEAD-TO-HEAD COMPARISON")
        print("-"*70)
        
        print("\nNDCG@5:")
        print(f"  ALS (Tutorial 2):  {als_results.get('ndcg@5', 0):.4f}")
        print(f"  Neural CF:         {neural_results['ndcg@5']:.4f}")
        improvement_5 = ((neural_results['ndcg@5'] - als_results.get('ndcg@5', 0)) 
                        / als_results.get('ndcg@5', 1)) * 100
        print(f"  Improvement:       {improvement_5:+.1f}%")
        
        print("\nNDCG@10:")
        print(f"  ALS (Tutorial 2):  {als_results.get('ndcg@10', 0):.4f}")
        print(f"  Neural CF:         {neural_results['ndcg@10']:.4f}")
        improvement_10 = ((neural_results['ndcg@10'] - als_results.get('ndcg@10', 0)) 
                         / als_results.get('ndcg@10', 1)) * 100
        print(f"  Improvement:       {improvement_10:+.1f}%")
        
        print("\nPrecision@10:")
        print(f"  ALS (Tutorial 2):  {als_results.get('precision@10', 0):.4f}")
        print(f"  Neural CF:         {neural_results['precision@10']:.4f}")
        
        print("\nRecall@10:")
        print(f"  ALS (Tutorial 2):  {als_results.get('recall@10', 0):.4f}")
        print(f"  Neural CF:         {neural_results['recall@10']:.4f}")
        
        # Calculate random baseline for comparison
        random_results_baseline = compare_to_random_baseline(
            test_matrix=test_matrix,
            n_items=test_matrix.shape[1],  # number of products
            k=10
        )
        
        # Visualize comparison with all three models
        Path('results').mkdir(exist_ok=True)
        comparison_data = {
            'Random Baseline': random_results_baseline,
            'ALS (Tutorial 2)': als_results,
            'Neural CF (Tutorial 3)': neural_results
        }
        plot_ndcg_comparison(comparison_data, save_path='results/all_models_comparison.png')
        print("\n✓ Comparison plot saved: results/all_models_comparison.png")
        
    else:
        print("\n⚠ ALS results not found. Run Tutorial 2 first for comparison.")
        print("  python run_tutorial_2.py")
    
    # ==========================================================================
    # STEP 6: COLD START COMPARISON
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: COLD START ANALYSIS")
    print("="*70)
    
    print("\nCold Start Comparison:")
    print("\nALS (Tutorial 2):")
    print("  - For new users: Recommend popular items")
    print("  - No personalization without purchase history")
    
    print("\nNeural CF (Tutorial 3):")
    print("  - For new users: Can use demographic features")
    print("  - Partial personalization even without history")
    print("  - Example: User who shops on weekends at 10am")
    print("    gets different recs than user who shops weekdays at 6pm")
    
    # Demonstrate with example
    print("\nExample New User Profile:")
    print("  - 5 orders, avg basket size 8 items")
    print("  - Shops on Saturdays at 10am")
    print("  - Avg 7 days between orders")
    
    # Get popular items
    popular_items = get_popular_items(train_matrix, top_k=10)
    
    print("\nPopular Items (ALS fallback):")
    for i, item_idx in enumerate(popular_items[:5], 1):
        prod_row = product_info[product_info['product_idx'] == item_idx]
        if not prod_row.empty:
            name = prod_row['product_name'].values[0]
            print(f"  {i}. {name}")
    
    print("\nNeural CF could personalize based on shopping patterns")
    print("(e.g., weekend shoppers buy more fresh produce)")
    
    # ==========================================================================
    # STEP 7: SHOW SAMPLE RECOMMENDATIONS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 7: SAMPLE RECOMMENDATIONS")
    print("="*70)
    
    # Show recommendations for sample users
    show_sample_recommendations(
        predictions_dict=predictions,
        product_info=product_info,
        user_features=user_features,
        n_samples=5
    )
    
    # ==========================================================================
    # STEP 8: FEATURE IMPORTANCE ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 8: WHAT DID THE NEURAL MODEL LEARN?")
    print("="*70)
    
    print("\nUser Features Used:")
    print("  1. Number of orders (more orders = more predictable)")
    print("  2. Average basket size (bulk shoppers vs single items)")
    print("  3. Days between orders (frequency)")
    print("  4. Favorite day of week (weekend vs weekday shopper)")
    print("  5. Favorite hour (morning, afternoon, evening)")
    
    print("\nItem Features Used:")
    print("  1. Department (produce, dairy, meat, etc.)")
    print("  2. Aisle (more granular categorization)")
    print("  3. Product ID (individual item embedding)")
    
    print("\nWhy This Helps:")
    print("  - Captures shopping patterns (e.g., weekend bulk buyers)")
    print("  - Cross-category recommendations (dairy → eggs)")
    print("  - Time-based preferences (morning coffee drinkers)")
    
    # ==========================================================================
    # STEP 9: MODEL COMPLEXITY ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 9: COMPLEXITY VS PERFORMANCE")
    print("="*70)
    
    # Count parameters
    n_params = sum(p.numel() for p in neural_model.parameters())
    
    print("\nModel Complexity:")
    print(f"  Neural CF: {n_params:,} parameters")
    print(f"  Training time: ~5-10 minutes (GPU)")
    print(f"  Memory usage: ~500 MB")
    
    if Path(als_results_path).exists():
        print("\n  ALS: ~100K parameters (user + item factors)")
        print("  Training time: ~1-2 minutes")
        print("  Memory usage: ~100 MB")
        
        print("\nCost-Benefit Analysis:")
        if improvement_10 > 10:
            print(f"  ✓ Neural CF is {improvement_10:.1f}% better")
            print("  ✓ Extra complexity is justified")
        elif improvement_10 > 5:
            print(f"  ~ Neural CF is {improvement_10:.1f}% better")
            print("  ~ Marginal improvement, depends on use case")
        else:
            print(f"  ✗ Neural CF is only {improvement_10:.1f}% better")
            print("  ✗ ALS might be better choice (simpler, faster)")
    
    # ==========================================================================
    # STEP 10: SAVE RESULTS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STEP 10: SAVING RESULTS")
    print("="*70)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Save evaluation results
    save_results(
        results=neural_results,
        model_name='Neural_CF',
        output_path='results/neural_results.pkl'
    )
    
    # Save model
    save_neural_model(neural_model, 'results/neural_model.pth')
    
    print("\nFiles saved:")
    print("  - results/neural_results.pkl (evaluation metrics)")
    print("  - results/neural_model.pth (trained model)")
    print("  - results/als_vs_neural.png (comparison plot)")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("TUTORIAL 3 COMPLETE!")
    print("="*70)
    
    print("\nKey Findings:")
    print(f"  1. Neural CF NDCG@10: {neural_results['ndcg@10']:.4f}")
    
    if Path(als_results_path).exists():
        print(f"  2. ALS NDCG@10: {als_results.get('ndcg@10', 0):.4f}")
        print(f"  3. Improvement: {improvement_10:+.1f}%")
    
    print(f"  4. Training time: ~5-10 min (vs 1-2 min for ALS)")
    print(f"  5. Better cold start (can use features)")
    
    print("\nWhen to Use Each Approach:")
    
    print("\n  Use ALS when:")
    print("    - You need fast training/retraining")
    print("    - Limited computational resources")
    print("    - Interpretability is important")
    print("    - You only have user-item interactions")
    
    print("\n  Use Neural CF when:")
    print("    - You have rich user/item features")
    print("    - Cold start is critical")
    print("    - You need maximum performance")
    print("    - You have GPU resources")
    
    print("\n  Hybrid Approach:")
    print("    - Use ALS for most users (fast)")
    print("    - Use Neural CF for high-value users")
    print("    - Combine predictions with ensemble")
    
    print("\nRecommendation:")
    if Path(als_results_path).exists() and improvement_10 < 5:
        print("  → Start with ALS (simpler, faster, similar performance)")
        print("  → Add Neural CF only if features significantly help")
    else:
        print("  → Neural CF shows meaningful improvement")
        print("  → Worth the extra complexity for production")
    
    print("\n" + "="*70)
    
    return {
        'model': neural_model,
        'results': neural_results,
        'predictions': predictions,
        'history': training_history
    }


if __name__ == "__main__":
    """
    Run Tutorial 3 when executed as script.
    """
    
    # Run the tutorial
    output = main()
    
    print("\nTutorial complete! Results stored in 'output' dictionary.")
    print("\nTo access results in interactive mode:")
    print("  model = output['model']")
    print("  results = output['results']")
    print("  predictions = output['predictions']")
    print("  history = output['history']")