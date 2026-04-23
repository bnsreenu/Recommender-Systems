"""
Instacart Recommender System - Neural Collaborative Filtering (Tutorial 3)

This module implements a Two-Tower neural network architecture for collaborative
filtering using PyTorch. Unlike ALS which only uses user/item IDs, this approach
can incorporate rich features (user demographics, item attributes, temporal info).

Architecture:
- User Tower: [user_id embedding + user features] → MLP → user vector (64-dim)
- Item Tower: [item_id embedding + item features] → MLP → item vector (64-dim)
- Score: dot product of user and item vectors → sigmoid → purchase probability

Key advantages over ALS:
- Can use side features (not just IDs)
- Better cold start (can recommend to new users using features)
- Non-linear patterns (MLP learns complex interactions)
- State-of-the-art performance

Trade-offs:
- Slower training (~5-10 min vs 1-2 min for ALS)
- More complex (requires tuning)
- Less interpretable
- May overfit with limited data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from tqdm import tqdm


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class RecommenderDataset(Dataset):
    """
    PyTorch Dataset for user-item interactions.
    
    Creates positive samples (actual purchases) and negative samples
    (random non-purchases) for training the neural network.
    """
    
    def __init__(self, interaction_matrix, user_features, item_features, 
                 negative_ratio=4):
        """
        Parameters:
        -----------
        interaction_matrix : scipy sparse matrix (users × items)
            Binary matrix where 1 = purchased, 0 = not purchased
        
        user_features : DataFrame
            User features (n_users × n_features)
        
        item_features : DataFrame
            Item features (n_items × n_features)
        
        negative_ratio : int
            Number of negative samples per positive sample
        """
        
        self.n_users = interaction_matrix.shape[0]
        self.n_items = interaction_matrix.shape[1]
        self.negative_ratio = negative_ratio
        
        # Convert to COO format for efficient sampling
        self.interactions = interaction_matrix.tocoo()
        
        # Prepare user features
        self.user_features = self._prepare_user_features(user_features)
        
        # Prepare item features
        self.item_features = self._prepare_item_features(item_features)
        
        # Create positive samples
        self.positive_samples = list(zip(
            self.interactions.row, 
            self.interactions.col
        ))
        
        # Prepare negative sampling lookup
        self.user_items = {}
        for user_idx, item_idx in self.positive_samples:
            if user_idx not in self.user_items:
                self.user_items[user_idx] = set()
            self.user_items[user_idx].add(item_idx)
    
    def _prepare_user_features(self, user_features_df):
        """
        Convert user features to tensor format.
        """
        # Select numerical features
        feature_cols = ['n_orders', 'avg_basket_size', 'avg_days_between_orders',
                       'favorite_dow', 'favorite_hour']
        
        # Handle missing columns gracefully
        available_cols = [col for col in feature_cols if col in user_features_df.columns]
        
        if not available_cols:
            # If no features available, create dummy features
            features = np.zeros((len(user_features_df), 5), dtype=np.float32)
        else:
            features = user_features_df[available_cols].values.astype(np.float32)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return torch.FloatTensor(features)
    
    def _prepare_item_features(self, item_features_df):
        """
        Convert item features to tensor format.
        """
        # Map categorical features
        if 'department' in item_features_df.columns:
            dept_mapping = {dept: i for i, dept in enumerate(item_features_df['department'].unique())}
        else:
            dept_mapping = {}
        
        if 'aisle' in item_features_df.columns:
            aisle_mapping = {aisle: i for i, aisle in enumerate(item_features_df['aisle'].unique())}
        else:
            aisle_mapping = {}
        
        features = []
        for _, row in item_features_df.sort_values('product_idx').iterrows():
            dept_id = dept_mapping.get(row.get('department', 0), 0)
            aisle_id = aisle_mapping.get(row.get('aisle', 0), 0)
            features.append([dept_id, aisle_id])
        
        return torch.LongTensor(features)
    
    def __len__(self):
        """
        Dataset size = positive samples × (1 + negative_ratio)
        """
        return len(self.positive_samples) * (1 + self.negative_ratio)
    
    def __getitem__(self, idx):
        """
        Get a training sample (user, item, label, features).
        
        Returns:
        --------
        user_id : int
        item_id : int
        user_feats : tensor
        item_feats : tensor
        label : float (1.0 for positive, 0.0 for negative)
        """
        
        # Determine if this is a positive or negative sample
        sample_idx = idx // (1 + self.negative_ratio)
        is_positive = (idx % (1 + self.negative_ratio)) == 0
        
        user_idx, pos_item_idx = self.positive_samples[sample_idx]
        
        if is_positive:
            # Positive sample: actual purchase
            item_idx = pos_item_idx
            label = 1.0
        else:
            # Negative sample: random non-purchased item
            item_idx = self._sample_negative_item(user_idx)
            label = 0.0
        
        # Get features
        user_feats = self.user_features[user_idx]
        item_feats = self.item_features[item_idx]
        
        return user_idx, item_idx, user_feats, item_feats, label
    
    def _sample_negative_item(self, user_idx):
        """
        Sample a random item that the user hasn't purchased.
        """
        while True:
            item_idx = np.random.randint(0, self.n_items)
            if item_idx not in self.user_items.get(user_idx, set()):
                return item_idx


# =============================================================================
# TWO-TOWER NEURAL NETWORK
# =============================================================================

class TwoTowerModel(nn.Module):
    """
    Two-Tower neural collaborative filtering model.
    
    Architecture:
    - User Tower: Embedding + Features → MLP → user_vector
    - Item Tower: Embedding + Features → MLP → item_vector
    - Score = user_vector · item_vector → sigmoid → probability
    """
    
    def __init__(self, n_users, n_items, n_departments, n_aisles,
                 embedding_dim=64, hidden_dims=[128, 64], 
                 user_feature_dim=5, dropout=0.2):
        """
        Parameters:
        -----------
        n_users : int
            Number of users
        
        n_items : int
            Number of items
        
        n_departments : int
            Number of product departments
        
        n_aisles : int
            Number of product aisles
        
        embedding_dim : int
            Dimension of learned embeddings
        
        hidden_dims : list
            Hidden layer sizes for MLPs
        
        user_feature_dim : int
            Number of user features
        
        dropout : float
            Dropout probability for regularization
        """
        
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # ===== USER TOWER =====
        
        # User ID embedding
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        
        # User MLP (embedding + features → user vector)
        user_input_dim = embedding_dim + user_feature_dim
        
        user_layers = []
        prev_dim = user_input_dim
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        user_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # ===== ITEM TOWER =====
        
        # Item ID embedding
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Item feature embeddings
        self.dept_embedding = nn.Embedding(n_departments, 16)
        self.aisle_embedding = nn.Embedding(n_aisles, 16)
        
        # Item MLP (embedding + features → item vector)
        item_input_dim = embedding_dim + 16 + 16  # id_emb + dept_emb + aisle_emb
        
        item_layers = []
        prev_dim = item_input_dim
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        item_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize embeddings and linear layers.
        """
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.dept_embedding.weight, std=0.01)
        nn.init.normal_(self.aisle_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids, user_features, item_features):
        """
        Forward pass through both towers.
        
        Parameters:
        -----------
        user_ids : tensor (batch_size,)
        item_ids : tensor (batch_size,)
        user_features : tensor (batch_size, user_feature_dim)
        item_features : tensor (batch_size, 2) [dept_id, aisle_id]
        
        Returns:
        --------
        scores : tensor (batch_size,)
            Purchase probability for each user-item pair
        """
        
        # ===== USER TOWER =====
        
        # Get user embedding
        user_emb = self.user_embedding(user_ids)  # (batch, embedding_dim)
        
        # Concatenate with user features
        user_input = torch.cat([user_emb, user_features], dim=1)
        
        # Pass through user MLP
        user_vector = self.user_tower(user_input)  # (batch, embedding_dim)
        
        # ===== ITEM TOWER =====
        
        # Get item embedding
        item_emb = self.item_embedding(item_ids)  # (batch, embedding_dim)
        
        # Get item feature embeddings
        dept_emb = self.dept_embedding(item_features[:, 0])  # (batch, 16)
        aisle_emb = self.aisle_embedding(item_features[:, 1])  # (batch, 16)
        
        # Concatenate all item features
        item_input = torch.cat([item_emb, dept_emb, aisle_emb], dim=1)
        
        # Pass through item MLP
        item_vector = self.item_tower(item_input)  # (batch, embedding_dim)
        
        # ===== SCORE COMPUTATION =====
        
        # Dot product of user and item vectors
        scores = (user_vector * item_vector).sum(dim=1)  # (batch,)
        
        # Sigmoid to get probability
        scores = torch.sigmoid(scores)
        
        return scores
    
    def get_user_embeddings(self, user_ids, user_features, device='cpu'):
        """
        Get user vectors for a batch of users.
        
        Used for inference: precompute user vectors once.
        """
        
        with torch.no_grad():
            user_ids = torch.LongTensor(user_ids).to(device)
            user_features = torch.FloatTensor(user_features).to(device)
            
            user_emb = self.user_embedding(user_ids)
            user_input = torch.cat([user_emb, user_features], dim=1)
            user_vectors = self.user_tower(user_input)
        
        return user_vectors.cpu().numpy()
    
    def get_item_embeddings(self, item_ids, item_features, device='cpu'):
        """
        Get item vectors for a batch of items.
        
        Used for inference: precompute item vectors once.
        """
        
        with torch.no_grad():
            item_ids = torch.LongTensor(item_ids).to(device)
            item_features = torch.LongTensor(item_features).to(device)
            
            item_emb = self.item_embedding(item_ids)
            dept_emb = self.dept_embedding(item_features[:, 0])
            aisle_emb = self.aisle_embedding(item_features[:, 1])
            
            item_input = torch.cat([item_emb, dept_emb, aisle_emb], dim=1)
            item_vectors = self.item_tower(item_input)
        
        return item_vectors.cpu().numpy()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_neural(train_matrix, user_features, item_features,
                 n_epochs=10, batch_size=1024, lr=0.001,
                 embedding_dim=64, hidden_dims=[128, 64],
                 negative_ratio=4, use_gpu=True):
    """
    Train neural collaborative filtering model.
    
    Parameters:
    -----------
    train_matrix : scipy sparse matrix
        User-item interaction matrix
    
    user_features : DataFrame
        User features
    
    item_features : DataFrame
        Item features
    
    n_epochs : int
        Number of training epochs
    
    batch_size : int
        Batch size for training
    
    lr : float
        Learning rate
    
    embedding_dim : int
        Embedding dimension
    
    hidden_dims : list
        Hidden layer sizes
    
    negative_ratio : int
        Negative samples per positive sample
    
    use_gpu : bool
        Use GPU if available
    
    Returns:
    --------
    model : TwoTowerModel
        Trained model
    
    history : dict
        Training history (loss per epoch)
    """
    
    print("\n" + "="*70)
    print("TRAINING NEURAL COLLABORATIVE FILTERING MODEL")
    print("="*70)
    
    # Device configuration
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = RecommenderDataset(
        train_matrix, user_features, item_features,
        negative_ratio=negative_ratio
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Batches per epoch: {len(dataloader):,}")
    
    # Initialize model
    print("\nInitializing model...")
    n_users = train_matrix.shape[0]
    n_items = train_matrix.shape[1]
    n_departments = item_features['department'].nunique()
    n_aisles = item_features['aisle'].nunique()
    
    model = TwoTowerModel(
        n_users=n_users,
        n_items=n_items,
        n_departments=n_departments,
        n_aisles=n_aisles,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        user_feature_dim=5,
        dropout=0.2
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Training loop
    print("\nTraining...")
    history = {'train_loss': []}
    
    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5  # Stop if no improvement for 5 epochs
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        
        for batch in progress_bar:
            user_ids, item_ids, user_feats, item_feats, labels = batch
            
            # Move to device
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            user_feats = user_feats.to(device)
            item_feats = item_feats.to(device)
            labels = labels.float().to(device)  # Convert to float32
            
            # Forward pass
            predictions = model(user_ids, item_ids, user_feats, item_feats)
            
            # Compute loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Average loss for epoch
        avg_loss = epoch_loss / len(dataloader)
        history['train_loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss - 0.001:  # Improvement threshold
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"  No improvement for {early_stopping_patience} epochs")
            print(f"  Best loss: {best_loss:.4f}")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return model, history


def predict_neural(model, user_features, item_features, user_ids=None, 
                   k=10, batch_size=1024, use_gpu=True):
    """
    Generate top-K recommendations using trained neural model.
    
    For efficiency, precomputes all item vectors once, then computes
    user-item scores via dot products.
    
    Parameters:
    -----------
    model : TwoTowerModel
        Trained model
    
    user_features : DataFrame
        User features
    
    item_features : DataFrame
        Item features
    
    user_ids : list or None
        User indices to predict for. If None, predicts for all users.
    
    k : int
        Number of recommendations per user
    
    batch_size : int
        Batch size for inference
    
    use_gpu : bool
        Use GPU if available
    
    Returns:
    --------
    predictions : dict
        {user_idx: [list of k product indices]}
    
    scores : dict
        {user_idx: [list of k scores]}
    """
    
    print("\nGenerating recommendations with neural model...")
    
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Prepare user features
    feature_cols = ['n_orders', 'avg_basket_size', 'avg_days_between_orders', 
                    'favorite_dow', 'favorite_hour']
    available_cols = [col for col in feature_cols if col in user_features.columns]
    
    if not available_cols:
        # Create dummy features if none available
        user_feat_array = np.zeros((len(user_features), 5), dtype=np.float32)
    else:
        user_feat_array = user_features[available_cols].values.astype(np.float32)
    
    # Normalize
    user_feat_array = (user_feat_array - user_feat_array.mean(axis=0)) / (user_feat_array.std(axis=0) + 1e-8)
    
    # Prepare item features
    if 'department' in item_features.columns:
        dept_mapping = {dept: i for i, dept in enumerate(item_features['department'].unique())}
    else:
        dept_mapping = {}
    
    if 'aisle' in item_features.columns:
        aisle_mapping = {aisle: i for i, aisle in enumerate(item_features['aisle'].unique())}
    else:
        aisle_mapping = {}
    
    item_feat_array = []
    for _, row in item_features.sort_values('product_idx').iterrows():
        dept_id = dept_mapping.get(row.get('department', 0), 0)
        aisle_id = aisle_mapping.get(row.get('aisle', 0), 0)
        item_feat_array.append([dept_id, aisle_id])
    item_feat_array = np.array(item_feat_array)
    
    # Precompute all item vectors (efficient for multiple users)
    print("  Precomputing item vectors...")
    all_item_ids = np.arange(len(item_features))
    item_vectors = model.get_item_embeddings(all_item_ids, item_feat_array, device)
    
    # Default to all users
    if user_ids is None:
        user_ids = list(range(len(user_features)))
    
    print(f"  Computing scores for {len(user_ids):,} users...")
    
    predictions = {}
    scores_dict = {}
    
    # Process users in batches
    for i in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_user_feats = user_feat_array[batch_user_ids]
        
        # Get user vectors
        user_vectors = model.get_user_embeddings(batch_user_ids, batch_user_feats, device)
        
        # Compute scores: user_vectors @ item_vectors.T
        batch_scores = user_vectors @ item_vectors.T  # (batch_users, n_items)
        
        # Get top K for each user
        for j, user_idx in enumerate(batch_user_ids):
            user_scores = batch_scores[j]
            top_k_indices = np.argsort(user_scores)[::-1][:k]
            top_k_scores = user_scores[top_k_indices]
            
            predictions[user_idx] = top_k_indices.tolist()
            scores_dict[user_idx] = top_k_scores.tolist()
    
    print(f"  Generated {k} recommendations per user")
    
    return predictions, scores_dict


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def save_neural_model(model, filepath):
    """
    Save trained neural model to disk.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_users': model.user_embedding.num_embeddings,
            'n_items': model.item_embedding.num_embeddings,
            'embedding_dim': model.embedding_dim
        }
    }, filepath)
    
    print(f"\nNeural model saved to: {filepath}")


def load_neural_model(filepath, n_departments, n_aisles):
    """
    Load saved neural model from disk.
    """
    checkpoint = torch.load(filepath)
    config = checkpoint['model_config']
    
    model = TwoTowerModel(
        n_users=config['n_users'],
        n_items=config['n_items'],
        n_departments=n_departments,
        n_aisles=n_aisles,
        embedding_dim=config['embedding_dim']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nNeural model loaded from: {filepath}")
    
    return model


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example showing how to train and use the neural model.
    For full tutorial, run: python run_tutorial_3.py
    """
    
    print("Neural CF Module - Example Usage")
    print("="*70)
    print("\nThis module contains the two-tower neural network for recommendations.")
    print("For full tutorial, run: python run_tutorial_3.py")
    print("\nKey components:")
    print("  - TwoTowerModel: Neural architecture")
    print("  - train_neural(): Training function")
    print("  - predict_neural(): Generate recommendations")