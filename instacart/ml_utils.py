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

class SVDRecommender:
    def __init__(self, n_components=50):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
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
            # Fit SVD model
            print("\nTraining SVD model...")
            self.svd.fit(train_matrix)
            
            # Transform the training data and store with user indices
            self.transformed_matrix = self.svd.transform(train_matrix)
            self.normalized_matrix = normalize(self.transformed_matrix)
            self.train_user_indices = np.array(self.user_ids)[mask]
            
            print("Model training completed successfully")

            # Calculate MSE and RMSE
            print("\nCalculating prediction error metrics...")
            # Transform test data
            test_transformed = self.svd.transform(self.test_matrix)
            # Reconstruct test data
            reconstructed = self.svd.inverse_transform(test_transformed)
            
            # Calculate MSE and RMSE
            mse = np.mean((self.test_matrix.toarray() - reconstructed) ** 2)
            rmse = np.sqrt(mse)
            
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")

            # After training, calculate and store metrics
            print("\nCalculating SVD model metrics...")
            metrics = self.calculate_metrics()
            self.stored_metrics = metrics
            
            print(f"SVD Model trained successfully with metrics:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            
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
            
    def _add_diversity(self, recommendations, diversity_weight):
        """Add diversity to recommendations based on department distribution"""
        try:
            # Load product data with department information
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            departments_df = pd.read_csv('instacart/static/csv/departments.csv')
            
            # Merge with departments
            products_df = products_df.merge(departments_df[['department_id', 'department']], on='department_id')
            
            # Get department distribution
            dept_dist = {}
            for rec in recommendations:
                aisle_id = rec['aisle_id']
                # Get department for this aisle
                dept = products_df[products_df['aisle_id'] == aisle_id]['department'].iloc[0]
                dept_dist[dept] = dept_dist.get(dept, 0) + 1
                
            # Adjust scores based on department diversity
            for rec in recommendations:
                aisle_id = rec['aisle_id']
                dept = products_df[products_df['aisle_id'] == aisle_id]['department'].iloc[0]
                diversity_penalty = (dept_dist[dept] / len(recommendations)) * diversity_weight
                rec['similarity_score'] *= (1 - diversity_penalty)
                # Add department info to recommendation
                rec['department'] = dept
                
            return sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            print(f"Error adding diversity: {str(e)}")
            return recommendations

    def calculate_metrics(self, threshold=0.5):
        """Calculate model performance metrics using test data"""
        if self.normalized_matrix is None or self.test_matrix is None:
            return None
        
        try:
            # Make predictions on test set
            test_predictions = self.svd.transform(self.test_matrix)
            
            # Project back to original space
            reconstructed = self.svd.inverse_transform(test_predictions)
            predicted = (reconstructed >= threshold).astype(int)
            
            # Convert sparse matrix to dense for calculations
            actual = self.test_matrix.toarray()
            
            # Calculate metrics
            true_positives = np.sum((actual > 0) & (predicted > 0))
            false_positives = np.sum((actual == 0) & (predicted > 0))
            true_negatives = np.sum((actual == 0) & (predicted == 0))
            false_negatives = np.sum((actual > 0) & (predicted == 0))
            
            # Calculate final metrics
            total_predictions = true_positives + true_negatives + false_positives + false_negatives
            accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'true_negatives': int(true_negatives),
                'false_negatives': int(false_negatives)
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None

    def save_model(self, filepath='instacart/static/model/svd_model.pkl'):
        """Save SVD model and metrics"""
        if self.normalized_matrix is None:
            print("No model to save. Train the model first.")
            return False
        
        Path('instacart/static/model').mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'svd': self.svd,
                'normalized_matrix': self.normalized_matrix,
                'transformed_matrix': self.transformed_matrix,
                'aisle_names': self.aisle_names,
                'user_ids': self.user_ids,
                'aisle_ids': self.aisle_ids,
                'user_to_idx': self.user_to_idx,
                'aisle_to_idx': self.aisle_to_idx,
                'idx_to_aisle': self.idx_to_aisle,
                'train_user_indices': self.train_user_indices,
                'stored_metrics': getattr(self, 'stored_metrics', None),  # Save stored metrics
                'test_matrix': self.test_matrix  # Add test matrix to saved data
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
                model.svd = data['svd']
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
            
            # Adjust n_components based on the number of features
            n_features = min(matrix.shape[0], matrix.shape[1])
            adjusted_n_components = min(n_components, n_features - 1)  # Ensure n_components < n_features
            
            # Create and train new SVD model
            cluster_svd = TruncatedSVD(n_components=adjusted_n_components, random_state=42)
            cluster_svd.fit(matrix)
            
            # Transform the data
            transformed_matrix = cluster_svd.transform(matrix)
            normalized_matrix = normalize(transformed_matrix)
            
            return {
                'svd': cluster_svd,
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

class SVDProductRecommender:
    def __init__(self, n_components=50):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.product_names = {}
        self.metrics = None
        self.normalized_matrix = None
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
            orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
            
            # Create product names dictionary
            self.product_names = dict(zip(products_df.product_id, products_df.product_name))
            
            return merged_df, orders_df, products_df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None

    def train_model(self, test_size=0.2, random_state=42):
        """Train SVD model for product recommendations using scikit-learn"""
        merged_df, orders_df, products_df = self.load_data()
        if merged_df is None:
            return False

        print("\nPreparing user-product interaction matrix...")
        
        # Create user-product interaction matrix
        user_product_data = (merged_df
            .merge(orders_df[['order_id', 'user_id']], on='order_id')
            .groupby(['user_id', 'product_id'])
            .size()
            .reset_index(name='interaction_count')
        )
        print(user_product_data)

        # Create binary ratings (1 if product was ordered, 0 otherwise)
        user_product_data['rating'] = user_product_data['interaction_count'].apply(lambda x: 1 if x > 0 else 0)

        # Create sparse matrix
        self.user_ids = sorted(user_product_data['user_id'].unique())
        self.product_ids = sorted(user_product_data['product_id'].unique())
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.product_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        self.idx_to_product = {idx: pid for pid, idx in self.product_to_idx.items()}

        matrix = csr_matrix(
            (user_product_data['rating'].astype(np.float32),
             (user_product_data['user_id'].map(self.user_to_idx),
              user_product_data['product_id'].map(self.product_to_idx))),
            shape=(len(self.user_ids), len(self.product_ids))
        )

        # Split data for evaluation
        np.random.seed(random_state)
        mask = np.random.rand(matrix.shape[0]) >= test_size
        train_matrix = matrix[mask]
        test_matrix = matrix[~mask]

        print(f"Training matrix shape: {train_matrix.shape}")
        print(f"Test matrix shape: {test_matrix.shape}")

        try:
            # Train the model
            print("\nTraining SVD model...")
            self.svd.fit(train_matrix)
            
            # Transform the training data
            self.transformed_matrix = self.svd.transform(train_matrix)
            self.normalized_matrix = normalize(self.transformed_matrix)
            
            # Make predictions on test set
            test_transformed = self.svd.transform(test_matrix)
            reconstructed = self.svd.inverse_transform(test_transformed)
            
            # Calculate metrics
            print("\nCalculating model metrics...")
            
            # Convert to binary predictions
            y_true = test_matrix.toarray()
            y_pred = (reconstructed >= 0.5).astype(int)
            
            # Calculate error metrics
            rmse = np.sqrt(mean_squared_error(y_true, reconstructed))
            mae = mean_absolute_error(y_true, reconstructed)
            
            # Calculate classification metrics
            acc = accuracy_score(y_true.flatten(), y_pred.flatten())
            prec = precision_score(y_true.flatten(), y_pred.flatten(), average='binary', zero_division=0)
            rec = recall_score(y_true.flatten(), y_pred.flatten(), average='binary', zero_division=0)
            f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='binary', zero_division=0)
            
            # Store metrics
            self.metrics = {
                'rmse': rmse,
                'mae': mae,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            }
            
            print(f"Model trained successfully with metrics:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Accuracy: {acc:.3f}")
            print(f"Precision: {prec:.3f}")
            print(f"Recall: {rec:.3f}")
            print(f"F1 Score: {f1:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def get_recommendations(self, user_id, n_recommendations=5):
        """Get product recommendations using SVD"""
        if self.normalized_matrix is None or not hasattr(self, 'user_to_idx'):
            return []

        try:
            if user_id not in self.user_to_idx:
                return []
                
            user_idx = self.user_to_idx[user_id]
            user_vector = self.normalized_matrix[user_idx]
            
            # Calculate similarity scores
            similarity_scores = np.dot(self.normalized_matrix, user_vector)
            
            # Get top recommendations
            top_indices = np.argsort(similarity_scores)[-n_recommendations*2:][::-1]
            
            recommendations = []
            for idx in top_indices:
                product_idx = idx % len(self.product_ids)
                product_id = self.idx_to_product[product_idx]
                score = float(similarity_scores[idx])
                recommendations.append({
                    'product_id': product_id,
                    'product_name': self.product_names.get(product_id, f"Product {product_id}"),
                    'similarity_score': score
                })
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def save_model(self, filepath='instacart/static/model/svd_product_model.pkl'):
        """Save SVD model and metrics"""
        if not hasattr(self, 'svd'):
            print("No model to save. Train the model first.")
            return False
        
        Path('instacart/static/model').mkdir(parents=True, exist_ok=True)
        
        model_info = {
            'svd': self.svd,
            'normalized_matrix': self.normalized_matrix,
            'product_names': self.product_names,
            'metrics': self.metrics,
            'user_to_idx': self.user_to_idx,
            'product_to_idx': self.product_to_idx,
            'idx_to_product': self.idx_to_product
        }
        
        joblib.dump(model_info, filepath)
        print(f"SVD product model saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath='instacart/static/model/svd_product_model.pkl'):
        """Load a saved SVD product model"""
        try:
            model_info = joblib.load(filepath)
            model = cls()
            model.svd = model_info['svd']
            model.normalized_matrix = model_info['normalized_matrix']
            model.product_names = model_info['product_names']
            model.metrics = model_info['metrics']
            model.user_to_idx = model_info['user_to_idx']
            model.product_to_idx = model_info['product_to_idx']
            model.idx_to_product = model_info['idx_to_product']
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

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
                # Train the model
                svd = TruncatedSVD(n_components=self.n_components, random_state=random_state)
                svd.fit(train_matrix)
                
                # Transform the training data
                transformed_matrix = svd.transform(train_matrix)
                normalized_matrix = normalize(transformed_matrix)
                
                # Make predictions on test set
                test_transformed = svd.transform(test_matrix)
                reconstructed = svd.inverse_transform(test_transformed)
                
                # Calculate metrics
                y_true = test_matrix.toarray()
                y_pred = (reconstructed >= 0.5).astype(int)
                
                # Calculate error metrics
                rmse = np.sqrt(mean_squared_error(y_true, reconstructed))
                mae = mean_absolute_error(y_true, reconstructed)
                
                # Calculate classification metrics
                acc = accuracy_score(y_true.flatten(), y_pred.flatten())
                prec = precision_score(y_true.flatten(), y_pred.flatten(), average='binary', zero_division=0)
                rec = recall_score(y_true.flatten(), y_pred.flatten(), average='binary', zero_division=0)
                f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='binary', zero_division=0)
                
                # Store model and metrics
                self.cluster_models[cluster_id] = {
                    'svd': svd,
                    'normalized_matrix': normalized_matrix,
                    'user_to_idx': user_to_idx,
                    'product_to_idx': product_to_idx,
                    'idx_to_product': idx_to_product,
                    'metrics': {
                        'rmse': rmse,
                        'mae': mae,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1
                    }
                }
                
                print(f"Cluster {cluster_id} model trained successfully with metrics:")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"Accuracy: {acc:.3f}")
                print(f"Precision: {prec:.3f}")
                print(f"Recall: {rec:.3f}")
                print(f"F1 Score: {f1:.3f}")
                
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

class ProductBasketModel:
    def __init__(self, batch_size=10000):
        self.rules = None
        self.product_names = {}
        self.batch_size = batch_size
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            products_df = pd.read_csv('instacart/static/csv/products.csv')
            merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
            orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
            
            # Create product names dictionary
            self.product_names = dict(zip(products_df.product_id, products_df.product_name))
            
            return merged_df, orders_df, products_df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None

    def process_batch(self, batch_data):
        """Process a batch of data to create product basket matrix"""
        basket_matrix = pd.crosstab(batch_data['order_id'], batch_data['product_id']) > 0
        return basket_matrix

    def train_model(self, min_support=0.01, min_lift=1.0, test_size=0.2, random_state=42):
        """Train product-based basket analysis model using batch processing"""
        merged_df, orders_df, products_df = self.load_data()
        if merged_df is None:
            return False

        print("\nCreating product-based basket data...")
        
        # Create basket data
        basket_data = (merged_df
            .merge(orders_df[['order_id', 'user_id']], on='order_id')
            [['order_id', 'product_id']]
            .drop_duplicates()
        )

        # Split data into train and test sets
        np.random.seed(random_state)
        order_ids = basket_data['order_id'].unique()
        train_mask = np.random.rand(len(order_ids)) >= test_size
        train_orders = order_ids[train_mask]
        test_orders = order_ids[~train_mask]

        # Process data in batches
        print("Processing data in batches...")
        train_matrices = []
        test_matrices = []
        
        for i in range(0, len(basket_data), self.batch_size):
            batch = basket_data.iloc[i:i + self.batch_size]
            
            # Split batch into train and test
            train_batch = batch[batch['order_id'].isin(train_orders)]
            test_batch = batch[batch['order_id'].isin(test_orders)]
            
            if not train_batch.empty:
                train_matrix = self.process_batch(train_batch)
                train_matrices.append(train_matrix)
            
            if not test_batch.empty:
                test_matrix = (self.process_batch(test_batch))
                test_matrices.append(test_matrix)
            
            print(f"Processed batch {i//self.batch_size + 1}")

        # Combine matrices
        if train_matrices:
            train_matrix = pd.concat(train_matrices)
            test_matrix = pd.concat(test_matrices) if test_matrices else None
        else:
            print("No training data available")
            return False

        print(f"Training set size: {len(train_matrix)} orders")
        print(f"Test set size: {len(test_matrix)} orders")

        try:
            # Generate frequent itemsets
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

                # Convert frozensets to lists
                rules['antecedents'] = rules['antecedents'].apply(list)
                rules['consequents'] = rules['consequents'].apply(list)

                # Calculate combined score
                rules['combined_score'] = rules['confidence'] * 0.7 + rules['lift'] * 0.3
                rules.sort_values('combined_score', ascending=False, inplace=True)
                
                self.test_matrix = test_matrix
                self.rules = rules

                # Calculate metrics
                metrics = self.calculate_metrics()
                self.stored_metrics = metrics
                
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

    def get_recommendations(self, product_id, n_recommendations=5):
        """Get product recommendations based on a given product"""
        if self.rules is None:
            return []
            
        try:
            recommendations = []
            
            # Find rules where the given product is in antecedents
            for _, rule in self.rules.iterrows():
                if product_id in rule['antecedents']:
                    for consequent in rule['consequents']:
                        recommendations.append({
                            'product_id': consequent,
                            'product_name': self.product_names.get(consequent, f"Product {consequent}"),
                            'confidence': float(rule['confidence']),
                            'lift': float(rule['lift']),
                            'support': float(rule['support']),
                            'score': float(rule['combined_score'])
                        })
            
            # Sort by score and remove duplicates
            recommendations = list({r['product_id']: r for r in recommendations}.values())
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def calculate_metrics(self):
        """Calculate model performance metrics"""
        if not hasattr(self, 'rules') or not hasattr(self, 'test_matrix'):
            print("Warning: Missing rules or test data")
            return None
        
        try:
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for _, rule in self.rules.iterrows():
                antecedent_cols = rule['antecedents']
                consequent_cols = rule['consequents']
                
                antecedent_mask = self.test_matrix[antecedent_cols].all(axis=1)
                orders_with_antecedents = self.test_matrix[antecedent_mask]
                
                if len(orders_with_antecedents) > 0:
                    consequent_present = orders_with_antecedents[consequent_cols].any(axis=1)
                    true_positives += consequent_present.sum()
                    false_positives += (~consequent_present).sum()
                
                orders_without_antecedents = self.test_matrix[~antecedent_mask]
                if len(orders_without_antecedents) > 0:
                    consequent_present = orders_without_antecedents[consequent_cols].any(axis=1)
                    false_negatives += consequent_present.sum()
                    true_negatives += (~consequent_present).sum()
            
            total = true_positives + false_positives + true_negatives + false_negatives
            if total == 0:
                return None
            
            metrics = {
                'accuracy': (true_positives + true_negatives) / total,
                'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
                'recall': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
                'f1_score': 2 * (true_positives / (true_positives + false_positives)) * (true_positives / (true_positives + false_negatives)) / ((true_positives / (true_positives + false_positives)) + (true_positives / (true_positives + false_negatives))) if (true_positives + false_positives) > 0 and (true_positives + false_negatives) > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None

    def save_model(self, filepath='instacart/static/model/product_basket_model.pkl'):
        """Save the trained model"""
        if self.rules is None:
            print("No rules to save")
            return False
            
        Path('instacart/static/model').mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'product_names': self.product_names,
                'stored_metrics': getattr(self, 'stored_metrics', None),
                'test_matrix': self.test_matrix
            }, f)
        print(f"Model saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath='instacart/static/model/product_basket_model.pkl'):
        """Load the saved model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                model = cls()
                model.rules = data['rules']
                model.product_names = data['product_names']
                model.stored_metrics = data.get('stored_metrics', None)
                model.test_matrix = data.get('test_matrix')
                return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

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
                .head(2000)
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
            
            for _, rule in rules.iterrows():
                antecedent_cols = rule['antecedents']
                consequent_cols = rule['consequents']
                
                antecedent_mask = test_matrix[antecedent_cols].all(axis=1)
                orders_with_antecedents = test_matrix[antecedent_mask]
                
                if len(orders_with_antecedents) > 0:
                    consequent_present = orders_with_antecedents[consequent_cols].any(axis=1)
                    true_positives += consequent_present.sum()
                    false_positives += (~consequent_present).sum()
                
                orders_without_antecedents = test_matrix[~antecedent_mask]
                if len(orders_without_antecedents) > 0:
                    consequent_present = orders_without_antecedents[consequent_cols].any(axis=1)
                    false_negatives += consequent_present.sum()
                    true_negatives += (~consequent_present).sum()
            
            total = true_positives + false_positives + true_negatives + false_negatives
            if total == 0:
                return None
            
            metrics = {
                'accuracy': (true_positives + true_negatives) / total,
                'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
                'recall': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
                'f1_score': 2 * (true_positives / (true_positives + false_positives)) * (true_positives / (true_positives + false_negatives)) / ((true_positives / (true_positives + false_positives)) + (true_positives / (true_positives + false_negatives))) if (true_positives + false_positives) > 0 and (true_positives + false_negatives) > 0 else 0
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