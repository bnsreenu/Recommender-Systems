"""
Data Preparation Script for Instacart Recommender Tutorials

This script prepares the Instacart dataset for Tutorials 2 and 3.
Run this ONCE before starting the tutorials.

Input: Raw Instacart CSV files (download from Kaggle)
Output: Processed subset (20K users × 800 products) ready for training

Usage:
    python prepare_data.py

The script will:
1. Load raw Instacart data from data/raw/
2. Filter to active users (5-20 orders)
3. Select top 800 most popular products
4. Create train/test split (80/20)
5. Generate user features for neural model
6. Save processed files to data/processed/
"""

import sys
from pathlib import Path

# Import the data preparation function from utils
from utils import load_and_prepare_data


def main():
    """
    Main data preparation workflow.
    """
    
    print("\n" + "="*70)
    print("INSTACART DATA PREPARATION FOR TUTORIALS")
    print("="*70)
    
    # Define paths
    raw_data_path = 'data/raw'
    processed_data_path = 'data/processed'
    
    # Check if raw data exists
    raw_path = Path(raw_data_path)
    if not raw_path.exists():
        print("\n" + "="*70)
        print("ERROR: Raw data directory not found!")
        print("="*70)
        print(f"\nExpected location: {raw_data_path}/")
        print("\nRequired files:")
        print("  - orders.csv")
        print("  - order_products__train.csv")
        print("  - products.csv")
        print("  - aisles.csv")
        print("  - departments.csv")
        print("\nDownload Instructions:")
        print("  1. Go to: https://www.kaggle.com/c/instacart-market-basket-analysis/data")
        print("  2. Download the dataset")
        print("  3. Extract files to data/raw/ directory")
        print("  4. Run this script again")
        return
    
    # Check for required files
    required_files = [
        'orders.csv',
        'order_products__train.csv',
        'products.csv',
        'aisles.csv',
        'departments.csv'
    ]
    
    missing_files = []
    for filename in required_files:
        if not (raw_path / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print("\n" + "="*70)
        print("ERROR: Missing required files!")
        print("="*70)
        print("\nMissing files:")
        for filename in missing_files:
            print(f"  - {filename}")
        print(f"\nPlease place all files in: {raw_data_path}/")
        return
    
    print("\n✓ All required files found")
    print(f"  Location: {raw_data_path}/")
    
    # Check if processed data already exists
    processed_path = Path(processed_data_path)
    if processed_path.exists() and any(processed_path.iterdir()):
        print("\n" + "="*70)
        print("WARNING: Processed data already exists!")
        print("="*70)
        print(f"\nLocation: {processed_data_path}/")
        print("\nOptions:")
        print("  1. Delete existing data and regenerate")
        print("  2. Keep existing data and skip preparation")
        
        response = input("\nDelete and regenerate? (y/n): ").strip().lower()
        
        if response != 'y':
            print("\nKeeping existing data. Exiting...")
            return
        
        # Remove existing processed data
        import shutil
        shutil.rmtree(processed_data_path)
        print("\n✓ Removed existing processed data")
    
    # Run data preparation
    print("\n" + "="*70)
    print("STARTING DATA PREPARATION")
    print("="*70)
    print("\nThis will take approximately 5-10 minutes...")
    print("Processing steps:")
    print("  1. Loading raw data")
    print("  2. Filtering users (5-20 orders)")
    print("  3. Selecting top 800 products")
    print("  4. Building interaction matrix")
    print("  5. Creating train/test split")
    print("  6. Generating user features")
    print("  7. Saving processed files")
    
    try:
        # Call the preparation function
        stats = load_and_prepare_data(
            raw_data_path=raw_data_path,
            processed_data_path=processed_data_path,
            n_users=80000, #previously 20000 (also try 40k or 80k)
            n_products=1500,  #previously 800 (also try 1200 or even 2000)
            min_orders=5,
            max_orders=20
        )
        
        # Success message
        print("\n" + "="*70)
        print("DATA PREPARATION SUCCESSFUL!")
        print("="*70)
        
        print("\nDataset Statistics:")
        print(f"  Users: {stats['n_users']:,}")
        print(f"  Products: {stats['n_products']:,}")
        print(f"  Interactions: {stats['n_interactions']:,}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")
        
        print(f"\nProcessed data saved to: {processed_data_path}/")
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Run Tutorial 2 (Classical ALS):")
        print("     python run_tutorial_2.py")
        print("\n2. Run Tutorial 3 (Neural CF):")
        print("     python run_tutorial_3.py")
        print("\n3. Compare results in results/ directory")
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR DURING DATA PREPARATION")
        print("="*70)
        print(f"\nError: {str(e)}")
        print("\nPlease check:")
        print("  1. All CSV files are in correct format")
        print("  2. Files are not corrupted")
        print("  3. Sufficient disk space available")
        print("  4. Sufficient RAM available (8GB+ recommended)")
        
        import traceback
        print("\nFull error trace:")
        traceback.print_exc()


if __name__ == "__main__":
    """
    Execute data preparation.
    """
    main()
