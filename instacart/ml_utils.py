import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pickle
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

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

    def train_model(self, min_support=0.05, min_lift=1.0, test_size=0.2, random_state=42):
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

                print(f"Number of rules generated: {len(rules)}")
                
                # Make sure to save the test data
                self.test_matrix = test_matrix
                self.rules = rules  # Save the rules

                print(f"Model trained successfully. Test matrix shape: {test_matrix.shape}")

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

    def get_recommendations(self, aisle_ids, n_recommendations=5):
        """Get aisle recommendations based on current aisles"""
        if self.rules is None:
            return []
            
        # Filter rules where antecedents match current aisles
        matching_rules = self.rules[
            self.rules['antecedents'].apply(
                lambda x: any(aid in x for aid in aisle_ids)
            )
        ]
        
        # Get unique consequent aisles that aren't in current aisles
        recommendations = []
        seen_aisles = set()
        
        for _, rule in matching_rules.iterrows():
            for aisle_id in rule['consequents']:
                if aisle_id not in aisle_ids and aisle_id not in seen_aisles:
                    recommendations.append({
                        'aisle_id': aisle_id,
                        'aisle_name': self.aisle_names.get(aisle_id, f"Aisle {aisle_id}"),
                        'confidence': float(rule['confidence']),
                        'lift': float(rule['lift']),
                        'support': float(rule['support'])
                    })
                    seen_aisles.add(aisle_id)
                    
                    if len(recommendations) >= n_recommendations:
                        return recommendations
                        
        return recommendations

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
        """Get aisle recommendations for a user"""
        if self.normalized_matrix is None or not hasattr(self, 'user_to_idx'):
            return []

        try:
            # Get user index
            if user_id not in self.user_to_idx:
                print(f"User {user_id} not in model data, using general popularity recommendations")
                # Return most popular aisles based on overall interaction counts
                aisle_popularity = self.test_matrix.sum(axis=0).A1  # Get column sums
                top_aisle_indices = np.argsort(aisle_popularity)[-n_recommendations:][::-1]
                
                recommendations = []
                for idx in top_aisle_indices:
                    aisle_id = self.idx_to_aisle[idx]
                    score = float(aisle_popularity[idx] / aisle_popularity.max())  # Normalize score
                    recommendations.append({
                        'aisle_id': aisle_id,
                        'aisle_name': self.aisle_names.get(aisle_id, f"Aisle {aisle_id}"),
                        'similarity_score': score
                    })
                return recommendations
            
            user_idx = self.user_to_idx[user_id]
            
            # Use the full matrix instead of just training data
            if hasattr(self, 'transformed_matrix'):
                user_vector = self.normalized_matrix[user_idx]
                similarity_scores = np.dot(self.normalized_matrix, user_vector)
            else:
                # If no transformed matrix available, fall back to popularity-based
                return self.get_popular_recommendations(n_recommendations)
            
            # Get top similar aisles
            top_indices = np.argsort(similarity_scores)[-n_recommendations:][::-1]
            
            recommendations = []
            for idx in top_indices:
                aisle_idx = idx % len(self.aisle_ids)  # Map back to aisle index
                aisle_id = self.idx_to_aisle[aisle_idx]
                score = float(similarity_scores[idx])
                recommendations.append({
                    'aisle_id': aisle_id,
                    'aisle_name': self.aisle_names.get(aisle_id, f"Aisle {aisle_id}"),
                    'similarity_score': score
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def get_popular_recommendations(self, n_recommendations=5):
        """Get recommendations based on overall popularity"""
        try:
            # Combine train and test matrices if available
            if hasattr(self, 'test_matrix'):
                full_matrix = self.test_matrix
            else:
                return []
            
            # Calculate aisle popularity
            aisle_popularity = full_matrix.sum(axis=0).A1
            top_aisle_indices = np.argsort(aisle_popularity)[-n_recommendations:][::-1]
            
            recommendations = []
            for idx in top_aisle_indices:
                aisle_id = self.idx_to_aisle[idx]
                score = float(aisle_popularity[idx] / aisle_popularity.max())
                recommendations.append({
                    'aisle_id': aisle_id,
                    'aisle_name': self.aisle_names.get(aisle_id, f"Aisle {aisle_id}"),
                    'similarity_score': score
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting popular recommendations: {str(e)}")
            return []

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