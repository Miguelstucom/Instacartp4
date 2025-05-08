import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pickle
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os

#MBA AISLE
class MarketBasketModel:
    def __init__(self):
        self.rules = None
        self.aisle_names = {}
        
    def load_data(self):
        """Carga los datos desde los archivos CSV"""
        try:
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
            orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
            aisles_df = pd.read_csv('instacart/static/csv/aisles.csv')
            
            # Merge aisles information into products
            products_df = products_df.merge(
                aisles_df[['aisle_id', 'aisle']], 
                on='aisle_id'
            )
            
            # Create aisle names dictionary
            self.aisle_names = dict(zip(aisles_df.aisle_id, aisles_df.aisle))
            
            return merged_df, orders_df, products_df
            
        except Exception as e:
            print(f"Error cargando datos: {str(e)}")
            return None, None, None

    def train_model(self, min_support=0.1, min_lift=1.2, test_size=0.2, random_state=42):
        """Train basket analysis model using aisle-to-aisle relationships with train-test split"""
        merged_df, orders_df, products_df = self.load_data()
        if merged_df is None:
            return False

        print("\nCreating aisle-based basket data...")
        
        # Create basket data more efficiently - do the merges once
        basket_data = (merged_df
            .merge(orders_df[['order_id', 'user_id']], on='order_id')
            .merge(products_df[['product_id', 'aisle_id']], on='product_id')
            [['order_id', 'aisle_id']]
            .drop_duplicates()
        )

        print(basket_data)

        # Create the full basket matrix first - more efficient than creating two separate ones
        basket_matrix = pd.crosstab(basket_data['order_id'], basket_data['aisle_id']) > 0
        print(basket_matrix)
        
        # Split using index instead of recreating matrices
        np.random.seed(random_state)
        order_ids = basket_matrix.index.values
        train_mask = np.random.rand(len(order_ids)) >= test_size
        
        # Split the existing matrix instead of creating new ones
        train_matrix = basket_matrix[train_mask]
        test_matrix = basket_matrix[~train_mask]
        
        print(f"Training set size: {len(train_matrix)} orders")
        print(f"Test set size: {len(test_matrix)} orders")

        try:
            # Generate frequent itemsets using training data
            print("\nGenerating frequent itemsets...")
            frequent_itemsets = apriori(
                train_matrix,
                min_support=min_support,
                use_colnames=True
            )
            print(f"Number of frequent itemsets found: {len(frequent_itemsets)}")

            if not frequent_itemsets.empty:
                print("\nGenerating association rules...")
                rules = association_rules(
                    frequent_itemsets,
                    metric="lift",
                    min_threshold=min_lift
                )

                # Convert frozensets to lists only once
                rules['antecedents'] = rules['antecedents'].apply(list)
                rules['consequents'] = rules['consequents'].apply(list)

                # Calculate combined score and sort
                rules['combined_score'] = rules['confidence'] * 0.7 + rules['lift'] * 0.3
                rules.sort_values('combined_score', ascending=False, inplace=True)
                print(rules)
                print(f"Number of rules generated: {len(rules)}")
                
                # Make sure to save the test data
                self.test_matrix = test_matrix
                self.rules = rules  # Save the rules

                print(f"Model trained successfully. Test matrix shape: {test_matrix.shape}")

                # Calculate MSE and RMSE
                print("\nCalculating prediction error metrics...")
                mse = 0
                total_predictions = 0
                
                for _, rule in rules.iterrows():
                    antecedent_cols = rule['antecedents']
                    consequent_cols = rule['consequents']
                    
                    # Find orders in test set that contain antecedents
                    antecedent_mask = test_matrix[antecedent_cols].all(axis=1)
                    orders_with_antecedents = test_matrix[antecedent_mask]
                    
                    if len(orders_with_antecedents) > 0:
                        # Check if these orders also contain consequents
                        consequent_present = orders_with_antecedents[consequent_cols].any(axis=1)
                        
                        # Calculate squared error for this rule
                        predicted = rule['confidence']
                        actual = consequent_present.mean()
                        mse += ((predicted - actual) ** 2).sum()
                        total_predictions += len(orders_with_antecedents)
                
                if total_predictions > 0:
                    mse = mse / total_predictions
                    rmse = np.sqrt(mse)
                    print(f"MSE: {mse:.4f}")
                    print(f"RMSE: {rmse:.4f}")
                else:
                    print("Warning: No predictions made for error calculation")

                # After training is complete, calculate and store metrics
                print("\nCalculating model metrics...")
                metrics = self.calculate_metrics()
                self.stored_metrics = metrics  # Store metrics as part of the model
                
                print(f"Model trained successfully with metrics:")
                print(f"Accuracy: {metrics['accuracy']:.3f}")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1 Score: {metrics['f1_score']:.3f}")
                
                return True
            else:
                print("No frequent itemsets found")
                return False

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def save_model(self, filepath='instacart/static/model/market_basket_model.pkl'):
        """Save the trained model and its metrics"""
        if self.rules is None:
            print("No rules to save. Train the model first.")
            return False
            
        Path('instacart/static/model').mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'aisle_names': self.aisle_names,
                'stored_metrics': getattr(self, 'stored_metrics', None),  # Save stored metrics
                'test_matrix': self.test_matrix  # Add test matrix to saved data
            }, f)
        print(f"Model and metrics saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath='instacart/static/model/market_basket_model.pkl'):
        """Load the saved model and its metrics"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                model = cls()
                model.rules = data['rules']
                model.aisle_names = data['aisle_names']
                model.stored_metrics = data.get('stored_metrics', None)  # Load stored metrics
                model.test_matrix = data.get('test_matrix')  # Load test matrix
                return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def get_recommendations(self, user_id, user_aisles, n_recommendations=5):
        """Get aisle recommendations based on user cluster and purchase history"""
        if self.rules is None:
            return []
            
        try:
            # Load user clusters
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['Cluster'].iloc[0]
            
            # Adjust recommendations based on cluster
            cluster_weights = {
                0: {'confidence': 0.6, 'lift': 0.5},  # Clientes frecuentes y activos
                1: {'confidence': 0.4, 'lift': 0.4},  # Clientes inactivos
                2: {'confidence': 0.3, 'lift': 0.8},  # Clientes Premium
                3: {'confidence': 0.5, 'lift': 0.6},  # Clientes regulares
            }
            
            # Get cluster-specific weights
            weights = cluster_weights.get(user_cluster, {'confidence': 0.5, 'lift': 0.5})
            
            # Use a dictionary to keep track of best scores for each aisle
            aisle_recommendations = {}
            
            # Calculate recommendations with cluster-specific scoring
            for _, rule in self.rules.iterrows():
                antecedents = rule['antecedents']
                if all(aid in user_aisles for aid in antecedents):
                    # Apply cluster weights to calculate score
                    score = (rule['confidence'] * weights['confidence'] + 
                            rule['lift'] * weights['lift'])
                    
                    for consequent in rule['consequents']:
                        # Remove the check for aisles not in user_aisles
                        # Instead, we'll include all aisles with good scores
                        if consequent not in aisle_recommendations or score > aisle_recommendations[consequent]['score']:
                            aisle_recommendations[consequent] = {
                                'aisle_id': consequent,
                                'aisle_name': self.aisle_names.get(consequent, f"Aisle {consequent}"),
                                'confidence': float(rule['confidence']),
                                'lift': float(rule['lift']),
                                'support': float(rule['support']),
                                'score': float(score),
                                'cluster': int(user_cluster),
                                'weights': weights,  # Include weights for transparency
                                'previously_bought': consequent in user_aisles  # Add flag for previously bought aisles
                            }
            
            # Convert dictionary to list and sort by score
            recommendations = list(aisle_recommendations.values())
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def calculate_metrics(self):
        """Calculate model performance metrics using test data"""
        if not hasattr(self, 'rules') or not hasattr(self, 'test_matrix'):
            print("Warning: Missing rules or test data. Please retrain the model.")
            return None
        
        try:
            # Print debug information
            print(f"Calculating metrics with {len(self.rules)} rules and test matrix shape {self.test_matrix.shape}")
            
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for _, rule in self.rules.iterrows():
                antecedent_cols = rule['antecedents']
                consequent_cols = rule['consequents']
                
                # Find orders in test set that contain antecedents
                antecedent_mask = self.test_matrix[antecedent_cols].all(axis=1)
                orders_with_antecedents = self.test_matrix[antecedent_mask]
                
                if len(orders_with_antecedents) > 0:
                    # Check if these orders also contain consequents
                    consequent_present = orders_with_antecedents[consequent_cols].any(axis=1)
                    
                    # Update metrics
                    true_positives += consequent_present.sum()
                    false_positives += (~consequent_present).sum()
                
                # Check for orders without antecedents
                orders_without_antecedents = self.test_matrix[~antecedent_mask]
                if len(orders_without_antecedents) > 0:
                    # Check if these orders contain consequents
                    consequent_present = orders_without_antecedents[consequent_cols].any(axis=1)
                    
                    # Update metrics
                    false_negatives += consequent_present.sum()
                    true_negatives += (~consequent_present).sum()
            
            # Add debug prints
            print(f"Calculated metrics: TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")
            
            total = true_positives + false_positives + true_negatives + false_negatives
            if total == 0:
                print("Warning: No predictions made")
                return None
            
            metrics = {
                'accuracy': (true_positives + true_negatives) / total,
                'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
                'recall': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives,
                'total_rules': len(self.rules),
                'avg_confidence': self.rules['confidence'].mean(),
                'avg_lift': self.rules['lift'].mean(),
                'avg_support': self.rules['support'].mean()
            }
            
            # Calculate F1 score
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None

    def get_aisle_products(self, user_id, aisle_id, n_products=3):
        """Get personalized product recommendations for a specific aisle"""
        try:
            # Load product data
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
            orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
            
            # Get user's purchase history
            user_history = merged_df.merge(
                orders_df[orders_df['user_id'] == user_id][['order_id']],
                on='order_id'
            ).merge(
                products_df[['product_id', 'product_name', 'aisle_id']],
                on='product_id'
            )
            
            # Get products from the specific aisle
            aisle_products = products_df[products_df['aisle_id'] == aisle_id]
            
            # Calculate product purchase frequency in this aisle
            product_freq = user_history[user_history['aisle_id'] == aisle_id]['product_id'].value_counts()
            
            # Get products user has bought in this aisle
            user_aisle_products = set(user_history[user_history['aisle_id'] == aisle_id]['product_id'])
            
            # Score products based on purchase frequency and user history
            product_scores = []
            for _, product in aisle_products.iterrows():
                score = 0
                if product['product_id'] in user_aisle_products:
                    # Higher score for previously purchased products
                    score = product_freq.get(product['product_id'], 0) * 2
                else:
                    # Base score for new products
                    score = 1
                    
                product_scores.append({
                    'product_id': product['product_id'],
                    'product_name': product['product_name'],
                    'score': score,
                    'previously_bought': product['product_id'] in user_aisle_products
                })
            
            # Sort by score and previously bought status
            product_scores.sort(key=lambda x: (x['score'], x['previously_bought']), reverse=True)
            
            return product_scores[:n_products]
            
        except Exception as e:
            print(f"Error getting aisle products: {str(e)}")
            return []

#SVD AISLE

class SVDRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.U = None  # Left singular vectors
        self.S = None  # Singular values
        self.Vh = None  # Right singular vectors
        self.normalized_matrix = None
        self.aisle_names = {}
        self.test_matrix = None
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
            orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
            aisles_df = pd.read_csv('instacart/static/csv/aisles.csv')
            
            self.aisle_names = dict(zip(aisles_df.aisle_id, aisles_df.aisle))
            
            return merged_df, orders_df, products_df, aisles_df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None, None

    def plot_explained_variance(self):
        """Plot individual and cumulative explained variance"""
        if self.S is None:
            print("No SVD model available. Train the model first.")
            return

        # Calculate explained variance
        explained_variance = (self.S ** 2) / (self.S ** 2).sum()
        cumulative_variance = np.cumsum(explained_variance)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot individual explained variance
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Explained Variance')
        ax1.set_title('Individual Explained Variance')
        
        # Plot cumulative explained variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        
        # Add grid and adjust layout
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('instacart/static/images/explained_variance.png')
        plt.close()
        
        # Print some statistics
        print(f"\nExplained Variance Statistics:")
        print(f"Total variance explained by {self.n_components} components: {cumulative_variance[self.n_components-1]:.2%}")
        print(f"Variance explained by first component: {explained_variance[0]:.2%}")
        print(f"Variance explained by last component: {explained_variance[self.n_components-1]:.2%}")

    def train_model(self, test_size=0.2, random_state=42):
        """Train SVD model and store metrics"""
        merged_df, orders_df, products_df, _ = self.load_data()
        if merged_df is None:
            return False

        print("\nPreparing user-aisle interaction matrix...")
        
        # Create user-aisle interaction matrix
        user_aisle_data = (merged_df
            .merge(orders_df[['order_id', 'user_id']], on='order_id')
            .merge(products_df[['product_id', 'aisle_id']], on='product_id')
            .groupby(['user_id', 'aisle_id'])
            .size()
            .reset_index(name='interaction_count')
        )

        # Store user and aisle mappings
        self.user_ids = sorted(user_aisle_data['user_id'].unique())
        self.aisle_ids = sorted(user_aisle_data['aisle_id'].unique())
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.aisle_to_idx = {aid: idx for idx, aid in enumerate(self.aisle_ids)}
        self.idx_to_aisle = {idx: aid for aid, idx in self.aisle_to_idx.items()}

        # Create sparse matrix with mapped indices
        matrix = csr_matrix(
            (user_aisle_data['interaction_count'],
             (user_aisle_data['user_id'].map(self.user_to_idx),
              user_aisle_data['aisle_id'].map(self.aisle_to_idx)))
        )

        # Split data for evaluation
        np.random.seed(random_state)
        mask = np.random.rand(matrix.shape[0]) >= test_size
        train_matrix = matrix[mask]
        self.test_matrix = matrix[~mask]

        print(f"Training matrix shape: {train_matrix.shape}")
        print(f"Test matrix shape: {self.test_matrix.shape}")

        try:
            # Convert sparse matrix to dense for SVD
            train_dense = train_matrix.toarray()
            
            # Perform SVD
            print("\nPerforming SVD decomposition...")
            self.U, self.S, self.Vh = np.linalg.svd(train_dense, full_matrices=False)
            
            # Keep only n_components
            self.U = self.U[:, :self.n_components]
            self.S = self.S[:self.n_components]
            self.Vh = self.Vh[:self.n_components, :]
            
            # Plot explained variance
            self.plot_explained_variance()
            
            # Transform the training data
            self.transformed_matrix = self.U * self.S
            self.normalized_matrix = normalize(self.transformed_matrix)
            self.train_user_indices = np.array(self.user_ids)[mask]
            
            print("Model training completed successfully")

            # Calculate MSE and RMSE
            print("\nCalculating prediction error metrics...")
            # Transform test data
            test_dense = self.test_matrix.toarray()
            test_transformed = test_dense @ self.Vh.T
            reconstructed = test_transformed @ self.Vh
            
            # Calculate MSE and RMSE
            mse = np.mean((test_dense - reconstructed) ** 2)
            rmse = np.sqrt(mse)
            
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")

            # Calculate classification metrics
            threshold = 0.5
            predicted = (reconstructed >= threshold).astype(int)
            true_positives = np.sum((test_dense > 0) & (predicted > 0))
            false_positives = np.sum((test_dense == 0) & (predicted > 0))
            true_negatives = np.sum((test_dense == 0) & (predicted == 0))
            false_negatives = np.sum((test_dense > 0) & (predicted == 0))
            
            total = true_positives + false_positives + true_negatives + false_negatives
            accuracy = (true_positives + true_negatives) / total if total > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Calculate ranking metrics
            print("\nCalculating ranking metrics...")
            ranking_metrics = calculate_ranking_metrics(test_dense, reconstructed)
            
            print("\nRanking Metrics:")
            print(f"Precision@{self.n_components}: {ranking_metrics['precision@k']:.4f}")
            print(f"Recall@{self.n_components}: {ranking_metrics['recall@k']:.4f}")
            print(f"NDCG@{self.n_components}: {ranking_metrics['ndcg@k']:.4f}")
            print(f"MAP@{self.n_components}: {ranking_metrics['map@k']:.4f}")
            print(f"Hit Rate@{self.n_components}: {ranking_metrics['hit_rate@k']:.4f}")

            # Store metrics
            self.stored_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'rmse': rmse,
                'mse': mse,
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'true_negatives': int(true_negatives),
                'false_negatives': int(false_negatives),
                'ranking_metrics': ranking_metrics
            }

            print("\nClassification Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def get_recommendations(self, user_id, n_recommendations=5):
        """Get recommendations using SVD and cluster information"""
        if self.normalized_matrix is None or not hasattr(self, 'user_to_idx'):
            return []

        try:
            # Load user clusters
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['Cluster'].iloc[0]
            
            # Get initial recommendations using the base model
            base_recommendations = []
            if user_id in self.user_to_idx:
                user_idx = self.user_to_idx[user_id]
                user_vector = self.normalized_matrix[user_idx]
                similarity_scores = np.dot(self.normalized_matrix, user_vector)
                top_indices = np.argsort(similarity_scores)[-n_recommendations*2:][::-1]
                
                for idx in top_indices:
                    aisle_idx = idx % len(self.aisle_ids)
                    aisle_id = self.idx_to_aisle[aisle_idx]
                    score = float(similarity_scores[idx])
                    base_recommendations.append({
                        'aisle_id': aisle_id,
                        'aisle_name': self.aisle_names.get(aisle_id, f"Aisle {aisle_id}"),
                        'similarity_score': score
                    })
            
            # Create cluster-specific SVD model
            cluster_model = self.create_cluster_svd_model(user_id, base_recommendations)
            if cluster_model is None:
                return base_recommendations[:n_recommendations]
            
            # Get recommendations from cluster model
            if user_id in cluster_model['user_to_idx']:
                user_idx = cluster_model['user_to_idx'][user_id]
                user_vector = cluster_model['normalized_matrix'][user_idx]
                similarity_scores = np.dot(cluster_model['normalized_matrix'], user_vector)
                
                # Get top recommendations
                top_indices = np.argsort(similarity_scores)[-n_recommendations:][::-1]
                recommendations = []
                
                for idx in top_indices:
                    aisle_idx = idx % len(cluster_model['aisle_ids'])
                    aisle_id = cluster_model['idx_to_aisle'][aisle_idx]
                    score = float(similarity_scores[idx])
                    recommendations.append({
                        'aisle_id': aisle_id,
                        'aisle_name': self.aisle_names.get(aisle_id, f"Aisle {aisle_id}"),
                        'similarity_score': score,
                        'cluster_specific': True
                    })
                
                return recommendations
            
            return base_recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []
            
    def calculate_ranking_metrics(self, y_true, y_pred, k=5):
        """Calculate ranking metrics for recommendations
        
        Args:
            y_true: Ground truth binary matrix
            y_pred: Predicted scores matrix
            k: Number of top items to consider
            
        Returns:
            Dictionary containing precision@k, recall@k, ndcg@k, map@k, hit_rate@k
        """
        metrics = {}
        
        # Get top k predictions for each user
        top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
        
        # Calculate metrics for each user
        precisions = []
        recalls = []
        ndcgs = []
        aps = []
        hits = []
        
        for i in range(len(y_true)):
            # Get true items for this user
            true_items = set(np.where(y_true[i] > 0)[0])
            if len(true_items) == 0:
                continue
            
            # Get predicted items
            pred_items = set(top_k_indices[i])
            
            # Calculate metrics
            # Precision@K
            precision = len(true_items & pred_items) / k
            precisions.append(precision)
            
            # Recall@K
            recall = len(true_items & pred_items) / len(true_items)
            recalls.append(recall)
            
            # Hit Rate@K
            hit = 1.0 if len(true_items & pred_items) > 0 else 0.0
            hits.append(hit)
            
            # NDCG@K
            dcg = 0
            idcg = 0
            for j, item in enumerate(top_k_indices[i]):
                if item in true_items:
                    dcg += 1 / np.log2(j + 2)  # j+2 because log2(1) = 0
            for j in range(min(len(true_items), k)):
                idcg += 1 / np.log2(j + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
            
            # MAP@K
            ap = 0
            correct = 0
            for j, item in enumerate(top_k_indices[i]):
                if item in true_items:
                    correct += 1
                    ap += correct / (j + 1)
            ap = ap / min(len(true_items), k) if len(true_items) > 0 else 0
            aps.append(ap)
        
        # Calculate average metrics
        metrics['precision@k'] = np.mean(precisions) if precisions else 0
        metrics['recall@k'] = np.mean(recalls) if recalls else 0
        metrics['ndcg@k'] = np.mean(ndcgs) if ndcgs else 0
        metrics['map@k'] = np.mean(aps) if aps else 0
        metrics['hit_rate@k'] = np.mean(hits) if hits else 0
        
        return metrics

    def create_cluster_svd_model(self, user_id, combined_recommendations, n_components=50):
        """Create a new SVD model specific to the user's cluster and combined recommendations"""
        try:
            # Load user clusters
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['Cluster'].iloc[0]
            
            # Get all users in the same cluster
            cluster_users = user_clusters[user_clusters['Cluster'] == user_cluster]['user_id'].tolist()
            
            # Get aisle IDs from combined recommendations
            recommended_aisle_ids = [rec['aisle_id'] for rec in combined_recommendations]
            
            # Load data
            merged_df, orders_df, products_df, _ = self.load_data()
            
            # Filter data for cluster users and recommended aisles
            cluster_data = merged_df.merge(
                orders_df[orders_df['user_id'].isin(cluster_users)][['order_id', 'user_id']],
                on='order_id'
            ).merge(
                products_df[products_df['aisle_id'].isin(recommended_aisle_ids)][['product_id', 'aisle_id']],
                on='product_id'
            )
            
            # Create user-aisle interaction matrix for the cluster
            user_aisle_data = (cluster_data
                .groupby(['user_id', 'aisle_id'])
                .size()
                .reset_index(name='interaction_count')
            )
            
            # Create mappings
            cluster_user_ids = sorted(user_aisle_data['user_id'].unique())
            cluster_aisle_ids = sorted(user_aisle_data['aisle_id'].unique())
            user_to_idx = {uid: idx for idx, uid in enumerate(cluster_user_ids)}
            aisle_to_idx = {aid: idx for idx, aid in enumerate(cluster_aisle_ids)}
            idx_to_aisle = {idx: aid for aid, idx in aisle_to_idx.items()}
            
            # Create sparse matrix
            matrix = csr_matrix(
                (user_aisle_data['interaction_count'],
                 (user_aisle_data['user_id'].map(user_to_idx),
                  user_aisle_data['aisle_id'].map(aisle_to_idx)))
            )
            
            # Convert to dense matrix for SVD
            dense_matrix = matrix.toarray()
            
            # Perform SVD
            U, S, Vh = np.linalg.svd(dense_matrix, full_matrices=False)
            
            # Keep only n_components
            U = U[:, :n_components]
            S = S[:n_components]
            Vh = Vh[:n_components, :]
            
            # Transform the data
            transformed_matrix = U * S
            normalized_matrix = normalize(transformed_matrix)
            
            return {
                'U': U,
                'S': S,
                'Vh': Vh,
                'normalized_matrix': normalized_matrix,
                'user_to_idx': user_to_idx,
                'aisle_to_idx': aisle_to_idx,
                'idx_to_aisle': idx_to_aisle,
                'user_ids': cluster_user_ids,
                'aisle_ids': cluster_aisle_ids
            }
            
        except Exception as e:
            print(f"Error creating cluster SVD model: {str(e)}")
            return None

    def save_model(self, filepath='instacart/static/model/svd_model.pkl'):
        """Save SVD model and metrics"""
        if self.normalized_matrix is None:
            print("No model to save. Train the model first.")
            return False
        
        Path('instacart/static/model').mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'U': self.U,
                'S': self.S,
                'Vh': self.Vh,
            'normalized_matrix': self.normalized_matrix,
                'transformed_matrix': self.transformed_matrix,
                'aisle_names': self.aisle_names,
                'user_ids': self.user_ids,
                'aisle_ids': self.aisle_ids,
            'user_to_idx': self.user_to_idx,
                'aisle_to_idx': self.aisle_to_idx,
                'idx_to_aisle': self.idx_to_aisle,
                'train_user_indices': self.train_user_indices,
                'stored_metrics': getattr(self, 'stored_metrics', None),
                'test_matrix': self.test_matrix
            }, f)
        print(f"SVD model saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath='instacart/static/model/svd_model.pkl'):
        """Load a saved SVD model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                model = cls()
                model.U = data['U']
                model.S = data['S']
                model.Vh = data['Vh']
                model.normalized_matrix = data['normalized_matrix']
                model.transformed_matrix = data['transformed_matrix']
                model.aisle_names = data['aisle_names']
                model.test_matrix = data['test_matrix']
                model.user_ids = data['user_ids']
                model.aisle_ids = data['aisle_ids']
                model.user_to_idx = data['user_to_idx']
                model.aisle_to_idx = data['aisle_to_idx']
                model.idx_to_aisle = data['idx_to_aisle']
                model.train_user_indices = data['train_user_indices']
                model.stored_metrics = data['stored_metrics']
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

#SVD PRODUCT CLUSTER

class SVDClusterRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.cluster_models = {}
        self.product_names = {}
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
            orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            
            # Create product names dictionary
            self.product_names = dict(zip(products_df.product_id, products_df.product_name))
            
            return merged_df, orders_df, products_df, user_clusters
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None, None

    def plot_cluster_explained_variance(self, cluster_id):
        """Plot individual and cumulative explained variance for a specific cluster"""
        if cluster_id not in self.cluster_models:
            print(f"No model available for cluster {cluster_id}. Train the model first.")
            return

        try:
            # Get SVD components for this cluster
            S = self.cluster_models[cluster_id]['S']
            
            # Calculate explained variance
            explained_variance = (S ** 2) / (S ** 2).sum()
            cumulative_variance = np.cumsum(explained_variance)
            
            # Find number of components needed for 95% variance
            components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot individual explained variance
            ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
            ax1.set_xlabel('Component')
            ax1.set_ylabel('Explained Variance')
            ax1.set_title(f'Individual Explained Variance - Cluster {cluster_id}')
            
            # Plot cumulative explained variance
            ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
            # Add horizontal line at 95%
            ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% Variance')
            # Add vertical line at components needed for 95%
            ax2.axvline(x=components_95, color='g', linestyle='--', alpha=0.5, 
                       label=f'Components needed: {components_95}')
            ax2.set_xlabel('Component')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.set_title(f'Cumulative Explained Variance - Cluster {cluster_id}')
            ax2.legend()
            
            # Add grid and adjust layout
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            os.makedirs('instacart/static/images', exist_ok=True)
            
            # Save the plot
            plot_path = f'instacart/static/images/explained_variance_cluster_{cluster_id}.png'
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Plot saved successfully to {plot_path}")
            
            # Print some statistics
            print(f"\nExplained Variance Statistics for Cluster {cluster_id}:")
            print(f"Total variance explained by {self.n_components} components: {cumulative_variance[self.n_components-1]:.2%}")
            print(f"Variance explained by first component: {explained_variance[0]:.2%}")
            print(f"Variance explained by last component: {explained_variance[self.n_components-1]:.2%}")
            print(f"Number of components needed for 95% variance: {components_95}")
            
            return True
            
        except Exception as e:
            print(f"Error generating plot for cluster {cluster_id}: {str(e)}")
            return False

    def train_model(self, test_size=0.2, random_state=42):
        """Train separate SVD models for each user cluster"""
        merged_df, orders_df, products_df, user_clusters = self.load_data()
        if merged_df is None:
            return False

        print("\nTraining SVD models for each cluster...")
        
        # Get unique clusters
        clusters = sorted(user_clusters['Cluster'].unique())
        
        for cluster_id in clusters:
            print(f"\n🔁 Training model for cluster {cluster_id}...")
            
            # Get users in this cluster
            cluster_users = user_clusters[user_clusters['Cluster'] == cluster_id]['user_id'].tolist()
            
            # Filter data for this cluster
            cluster_data = merged_df.merge(
                orders_df[orders_df['user_id'].isin(cluster_users)][['order_id', 'user_id']],
                on='order_id'
            )
            
            # Get top 2000 most purchased products in this cluster
            top_products = (cluster_data
                .groupby('product_id')
                .size()
                .reset_index(name='count')
                .sort_values('count', ascending=False)
                .head(4000)
            )
            
            # Filter data to only include top 2000 products
            cluster_data = cluster_data[cluster_data['product_id'].isin(top_products['product_id'])]
            
            # Create user-product interaction matrix
            user_product_data = (cluster_data
                .groupby(['user_id', 'product_id'])
                .size()
                .reset_index(name='interaction_count')
            )
            
            # Create binary ratings (1 if product was ordered, 0 otherwise)
            user_product_data['rating'] = user_product_data['interaction_count'].apply(lambda x: 1 if x > 0 else 0)
            
            # Create sparse matrix
            user_ids = sorted(user_product_data['user_id'].unique())
            product_ids = sorted(user_product_data['product_id'].unique())
            user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
            product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
            idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}
            
            matrix = csr_matrix(
                (user_product_data['rating'].astype(np.float32),
                 (user_product_data['user_id'].map(user_to_idx),
                  user_product_data['product_id'].map(product_to_idx))),
                shape=(len(user_ids), len(product_ids))
            )
            
            # Split data for evaluation
            np.random.seed(random_state)
            mask = np.random.rand(matrix.shape[0]) >= test_size
            train_matrix = matrix[mask]
            test_matrix = matrix[~mask]
            
            print(f"Cluster {cluster_id} - Training matrix shape: {train_matrix.shape}")
            print(f"Cluster {cluster_id} - Test matrix shape: {test_matrix.shape}")
            
            try:
                # Convert sparse matrix to dense for SVD
                train_dense = train_matrix.toarray()
                
                # Perform SVD
                print(f"\nPerforming SVD decomposition for cluster {cluster_id}...")
                U, S, Vh = np.linalg.svd(train_dense, full_matrices=False)
                
                # Keep only n_components
                U = U[:, :self.n_components]
                S = S[:self.n_components]
                Vh = Vh[:self.n_components, :]
                
                # Store model components
                self.cluster_models[cluster_id] = {
                    'U': U,
                    'S': S,
                    'Vh': Vh,
                    'normalized_matrix': None,  # Will be set after transformation
                    'user_to_idx': user_to_idx,
                    'product_to_idx': product_to_idx,
                    'idx_to_product': idx_to_product,
                    'metrics': None  # Will be set after metrics calculation
                }
                
                # Plot explained variance for this cluster
                print(f"\nGenerating explained variance plot for cluster {cluster_id}...")
                plot_success = self.plot_cluster_explained_variance(cluster_id)
                if not plot_success:
                    print(f"Warning: Failed to generate plot for cluster {cluster_id}")
                
                # Transform the training data
                transformed_matrix = U * S
                normalized_matrix = normalize(transformed_matrix)
                self.cluster_models[cluster_id]['normalized_matrix'] = normalized_matrix
                
                # Make predictions on test set
                test_dense = test_matrix.toarray()
                test_transformed = test_dense @ Vh.T
                reconstructed = test_transformed @ Vh
                
                # Calculate metrics
                y_true = test_dense
                y_pred = (reconstructed >= 0.5).astype(int)
                
                # Calculate error metrics
                rmse = np.sqrt(np.mean((y_true - reconstructed) ** 2))
                mae = np.mean(np.abs(y_true - reconstructed))
                
                # Calculate classification metrics
                true_positives = np.sum((y_true > 0) & (y_pred > 0))
                false_positives = np.sum((y_true == 0) & (y_pred > 0))
                true_negatives = np.sum((y_true == 0) & (y_pred == 0))
                false_negatives = np.sum((y_true > 0) & (y_pred == 0))
                
                total = true_positives + false_positives + true_negatives + false_negatives
                accuracy = (true_positives + true_negatives) / total if total > 0 else 0
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Calculate ranking metrics
                print(f"\nCalculating ranking metrics for cluster {cluster_id}...")
                ranking_metrics = calculate_ranking_metrics(test_dense, reconstructed)
                
                print(f"\nRanking Metrics for Cluster {cluster_id}:")
                print(f"Precision@{self.n_components}: {ranking_metrics['precision@k']:.4f}")
                print(f"Recall@{self.n_components}: {ranking_metrics['recall@k']:.4f}")
                print(f"NDCG@{self.n_components}: {ranking_metrics['ndcg@k']:.4f}")
                print(f"MAP@{self.n_components}: {ranking_metrics['map@k']:.4f}")
                print(f"Hit Rate@{self.n_components}: {ranking_metrics['hit_rate@k']:.4f}")
                
                # Store model and metrics
                self.cluster_models[cluster_id]['metrics'] = {
                        'rmse': rmse,
                        'mae': mae,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'ranking_metrics': ranking_metrics
                }
                
                print(f"Cluster {cluster_id} model trained successfully with metrics:")
                print(f"Accuracy: {accuracy:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1 Score: {f1:.3f}")
                print(f"Average Lift: {self.cluster_models[cluster_id]['metrics']['avg_lift']:.3f}")
                print(f"Average Confidence: {self.cluster_models[cluster_id]['metrics']['avg_confidence']:.3f}")
                print(f"Average Support: {self.cluster_models[cluster_id]['metrics']['avg_support']:.3f}")
                print(f"Total Rules: {self.cluster_models[cluster_id]['metrics']['total_rules']}")
                print(f"True Positives: {self.cluster_models[cluster_id]['metrics']['true_positives']}")
                print(f"False Positives: {self.cluster_models[cluster_id]['metrics']['false_positives']}")
                print(f"True Negatives: {self.cluster_models[cluster_id]['metrics']['true_negatives']}")
                print(f"False Negatives: {self.cluster_models[cluster_id]['metrics']['false_negatives']}")
                
            except Exception as e:
                print(f"Error training model for cluster {cluster_id}: {str(e)}")
                continue
        
        return len(self.cluster_models) > 0

    def get_recommendations(self, user_id, n_recommendations=5):
        """Get product recommendations using cluster-specific SVD model"""
        if not self.cluster_models:
            return []

        try:
            # Load user clusters
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['Cluster'].iloc[0]
            
            # Get cluster-specific model
            cluster_model = self.cluster_models.get(user_cluster)
            if cluster_model is None:
                return []
            
            if user_id not in cluster_model['user_to_idx']:
                return []
            
            user_idx = cluster_model['user_to_idx'][user_id]
            user_vector = cluster_model['normalized_matrix'][user_idx]
            
            # Calculate similarity scores
            similarity_scores = np.dot(cluster_model['normalized_matrix'], user_vector)
            
            # Get top recommendations
            top_indices = np.argsort(similarity_scores)[-n_recommendations*2:][::-1]
            
            recommendations = []
            for idx in top_indices:
                product_idx = idx % len(cluster_model['product_to_idx'])
                product_id = cluster_model['idx_to_product'][product_idx]
                score = float(similarity_scores[idx])
                recommendations.append({
                    'product_id': product_id,
                    'product_name': self.product_names.get(product_id, f"Product {product_id}"),
                    'similarity_score': score,
                    'cluster': int(user_cluster)
                })
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def save_model(self, filepath='instacart/static/model/svd_cluster_models.pkl'):
        """Save cluster-specific SVD models"""
        if not self.cluster_models:
            print("No models to save. Train the models first.")
            return False
        
        Path('instacart/static/model').mkdir(parents=True, exist_ok=True)
        
        model_info = {
            'cluster_models': self.cluster_models,
            'product_names': self.product_names,
            'n_components': self.n_components
        }
        
        joblib.dump(model_info, filepath)
        print(f"SVD cluster models saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath='instacart/static/model/svd_cluster_models.pkl'):
        """Load saved cluster-specific SVD models"""
        try:
            model_info = joblib.load(filepath)
            model = cls(n_components=model_info['n_components'])
            model.cluster_models = model_info['cluster_models']
            model.product_names = model_info['product_names']
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

#MBA PRODUCTS AISLE

class ClusterProductBasketModel:
    def __init__(self):
        self.cluster_models = {}  # Dictionary to store models for each cluster
        self.product_names = {}
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
            orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            
            # Create product names dictionary
            self.product_names = dict(zip(products_df.product_id, products_df.product_name))
            
            return merged_df, orders_df, products_df, user_clusters
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None, None

    def train_model(self, min_support=0.01, min_lift=1.0, test_size=0.2, random_state=42):
        """Train product-based basket analysis model for each cluster"""
        merged_df, orders_df, products_df, user_clusters = self.load_data()
        if merged_df is None:
            return False

        print("\nCreating cluster-based product basket data...")
        
        # Get unique clusters
        clusters = sorted(user_clusters['Cluster'].unique())
        
        for cluster_id in clusters:
            print(f"\n🔁 Training model for cluster {cluster_id}...")
            
            # Get users in this cluster
            cluster_users = user_clusters[user_clusters['Cluster'] == cluster_id]['user_id'].tolist()
            
            # Filter data for this cluster
            cluster_data = (merged_df
                .merge(orders_df[orders_df['user_id'].isin(cluster_users)][['order_id', 'user_id']], on='order_id')
                [['order_id', 'product_id']]
                .drop_duplicates()
            )
            
            if len(cluster_data) == 0:
                print(f"No data available for cluster {cluster_id}")
                continue

            # Get top 2000 most purchased products in this cluster
            top_products = (cluster_data
                .groupby('product_id')
                .size()
                .reset_index(name='count')
                .sort_values('count', ascending=False)
                .head(1500)
            )
            
            # Filter data to only include top 2000 products
            cluster_data = cluster_data[cluster_data['product_id'].isin(top_products['product_id'])]

            # Split data into train and test sets
            np.random.seed(random_state)
            order_ids = cluster_data['order_id'].unique()
            train_mask = np.random.rand(len(order_ids)) >= test_size
            train_orders = order_ids[train_mask]
            test_orders = order_ids[~train_mask]

            # Create basket matrices
            train_data = cluster_data[cluster_data['order_id'].isin(train_orders)]
            test_data = cluster_data[cluster_data['order_id'].isin(test_orders)]
            
            train_matrix = pd.crosstab(train_data['order_id'], train_data['product_id']) > 0
            test_matrix = pd.crosstab(test_data['order_id'], test_data['product_id']) > 0 if not test_data.empty else None

            print(f"Cluster {cluster_id} - Training set size: {len(train_matrix)} orders")
            print(f"Cluster {cluster_id} - Test set size: {len(test_matrix)} orders")

            try:
                # Generate frequent itemsets
                print(f"\nGenerating frequent itemsets for cluster {cluster_id}...")
                frequent_itemsets = apriori(
                    train_matrix,
                    min_support=min_support,
                    use_colnames=True
                )
                print(f"Number of frequent itemsets found for cluster {cluster_id}: {len(frequent_itemsets)}")

                if not frequent_itemsets.empty:
                    print(f"\nGenerating association rules for cluster {cluster_id}...")
                    rules = association_rules(
                        frequent_itemsets,
                        metric="lift",
                        min_threshold=min_lift
                    )

                    # Convert frozensets to lists
                    rules['antecedents'] = rules['antecedents'].apply(list)
                    rules['consequents'] = rules['consequents'].apply(list)

                    # Calculate combined score
                    rules['combined_score'] = rules['confidence'] * 0.7 + rules['lift'] * 0.3
                    rules.sort_values('combined_score', ascending=False, inplace=True)
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(rules, test_matrix)
                    
                    # Store model for this cluster
                    self.cluster_models[cluster_id] = {
                        'rules': rules,
                        'test_matrix': test_matrix,
                        'metrics': metrics
                    }
                    
                    print(f"Cluster {cluster_id} model trained successfully with metrics:")
                    print(f"Accuracy: {metrics['accuracy']:.3f}")
                    print(f"Precision: {metrics['precision']:.3f}")
                    print(f"Recall: {metrics['recall']:.3f}")
                    print(f"F1 Score: {metrics['f1_score']:.3f}")
                    print(f"Average Lift: {metrics['avg_lift']:.3f}")
                    print(f"Average Confidence: {metrics['avg_confidence']:.3f}")
                    print(f"Average Support: {metrics['avg_support']:.3f}")
                    print(f"Total Rules: {metrics['total_rules']}")
                    print(f"True Positives: {metrics['true_positives']}")
                    print(f"False Positives: {metrics['false_positives']}")
                    print(f"True Negatives: {metrics['true_negatives']}")
                    print(f"False Negatives: {metrics['false_negatives']}")
                else:
                    print(f"No frequent itemsets found for cluster {cluster_id}")

            except Exception as e:
                print(f"Error during training for cluster {cluster_id}: {str(e)}")
                continue
        
        return len(self.cluster_models) > 0

    def get_recommendations(self, product_id, user_id, n_recommendations=5):
        """Get product recommendations based on a given product and user's cluster"""
        if not self.cluster_models:
            return []
            
        try:
            # Get user's cluster
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['Cluster'].iloc[0]
            
            # Get model for user's cluster
            cluster_model = self.cluster_models.get(user_cluster)
            if cluster_model is None:
                return []
            
            recommendations = []
            
            # Find rules where the given product is in antecedents
            for _, rule in cluster_model['rules'].iterrows():
                if product_id in rule['antecedents']:
                    for consequent in rule['consequents']:
                        recommendations.append({
                            'product_id': consequent,
                            'product_name': self.product_names.get(consequent, f"Product {consequent}"),
                            'confidence': float(rule['confidence']),
                            'lift': float(rule['lift']),
                            'support': float(rule['support']),
                            'score': float(rule['combined_score']),
                            'cluster': int(user_cluster)
                        })
            
            # Sort by score and remove duplicates
            recommendations = list({r['product_id']: r for r in recommendations}.values())
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def calculate_metrics(self, rules, test_matrix):
        """Calculate model performance metrics"""
        if rules is None or test_matrix is None:
            return None
        
        try:
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            # Calculate average rule metrics
            avg_lift = rules['lift'].mean()
            avg_confidence = rules['confidence'].mean()
            avg_support = rules['support'].mean()
            total_rules = len(rules)
            
            for _, rule in rules.iterrows():
                antecedent_cols = rule['antecedents']
                consequent_cols = rule['consequents']
                
                # Find orders in test set that contain antecedents
                antecedent_mask = test_matrix[antecedent_cols].all(axis=1)
                orders_with_antecedents = test_matrix[antecedent_mask]
                
                if len(orders_with_antecedents) > 0:
                    # Check if these orders also contain consequents
                    consequent_present = orders_with_antecedents[consequent_cols].any(axis=1)
                    
                    # Update metrics
                    true_positives += consequent_present.sum()
                    false_positives += (~consequent_present).sum()
                
                # Check for orders without antecedents
                orders_without_antecedents = test_matrix[~antecedent_mask]
                if len(orders_without_antecedents) > 0:
                    # Check if these orders contain consequents
                    consequent_present = orders_without_antecedents[consequent_cols].any(axis=1)
                    
                    # Update metrics
                    false_negatives += consequent_present.sum()
                    true_negatives += (~consequent_present).sum()
            
            total = true_positives + false_positives + true_negatives + false_negatives
            if total == 0:
                return None
            
            # Calculate classification metrics
            accuracy = (true_positives + true_negatives) / total
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'avg_lift': avg_lift,
                'avg_confidence': avg_confidence,
                'avg_support': avg_support,
                'total_rules': total_rules,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None

    def save_model(self, filepath='instacart/static/model/cluster_product_basket_model.pkl'):
        """Save the trained models"""
        if not self.cluster_models:
            print("No models to save")
            return False
            
        Path('instacart/static/model').mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'cluster_models': self.cluster_models,
                'product_names': self.product_names
            }, f)
        print(f"Models saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath='instacart/static/model/cluster_product_basket_model.pkl'):
        """Load the saved models"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                model = cls()
                model.cluster_models = data['cluster_models']
                model.product_names = data['product_names']
                return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None 

def calculate_ranking_metrics(y_true, y_pred, k=5):
    """Calculate ranking metrics for recommendations
    
    Args:
        y_true: Ground truth binary matrix
        y_pred: Predicted scores matrix
        k: Number of top items to consider
        
    Returns:
        Dictionary containing precision@k, recall@k, ndcg@k, map@k, hit_rate@k
    """
    metrics = {}
    
    # Get top k predictions for each user
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    
    # Calculate metrics for each user
    precisions = []
    recalls = []
    ndcgs = []
    aps = []
    hits = []
    
    for i in range(len(y_true)):
        # Get true items for this user
        true_items = set(np.where(y_true[i] > 0)[0])
        if len(true_items) == 0:
            continue
            
        # Get predicted items
        pred_items = set(top_k_indices[i])
        
        # Calculate metrics
        # Precision@K
        precision = len(true_items & pred_items) / k
        precisions.append(precision)
        
        # Recall@K
        recall = len(true_items & pred_items) / len(true_items)
        recalls.append(recall)
        
        # Hit Rate@K
        hit = 1.0 if len(true_items & pred_items) > 0 else 0.0
        hits.append(hit)
        
        # NDCG@K
        dcg = 0
        idcg = 0
        for j, item in enumerate(top_k_indices[i]):
            if item in true_items:
                dcg += 1 / np.log2(j + 2)  # j+2 because log2(1) = 0
        for j in range(min(len(true_items), k)):
            idcg += 1 / np.log2(j + 2)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
        
        # MAP@K
        ap = 0
        correct = 0
        for j, item in enumerate(top_k_indices[i]):
            if item in true_items:
                correct += 1
                ap += correct / (j + 1)
        ap = ap / min(len(true_items), k) if len(true_items) > 0 else 0
        aps.append(ap)
    
    # Calculate average metrics
    metrics['precision@k'] = np.mean(precisions) if precisions else 0
    metrics['recall@k'] = np.mean(recalls) if recalls else 0
    metrics['ndcg@k'] = np.mean(ndcgs) if ndcgs else 0
    metrics['map@k'] = np.mean(aps) if aps else 0
    metrics['hit_rate@k'] = np.mean(hits) if hits else 0
    
    return metrics 