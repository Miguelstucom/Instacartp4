from django.core.management.base import BaseCommand
from instacart.ml_utils import SVDClusterRecommender
import os

class Command(BaseCommand):
    help = 'Train SVD recommendation models for each user cluster'

    def add_arguments(self, parser):
        parser.add_argument(
            '--components',
            type=int,
            default=50,
            help='Number of components for SVD'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Proportion of data to use for testing'
        )

    def handle(self, *args, **options):
        self.stdout.write("Starting SVD cluster model training...")
        
        # Create images directory if it doesn't exist
        os.makedirs('instacart/static/images', exist_ok=True)
        
        # Initialize and train the model
        model = SVDClusterRecommender(n_components=options['components'])
        success = model.train_model(test_size=options['test_size'])
        
        if success:
            # Save the trained model
            model.save_model()
            self.stdout.write(
                self.style.SUCCESS('Successfully trained and saved SVD cluster models')
            )
            
            # Print metrics for each cluster
            for cluster_id, cluster_model in model.cluster_models.items():
                metrics = cluster_model['metrics']
                self.stdout.write(f"\nMetrics for Cluster {cluster_id}:")
                self.stdout.write('-' * 30)
                self.stdout.write(f"RMSE: {metrics['rmse']:.4f}")
                self.stdout.write(f"MAE: {metrics['mae']:.4f}")
                self.stdout.write(f"Accuracy: {metrics['accuracy']:.4f}")
                self.stdout.write(f"Precision: {metrics['precision']:.4f}")
                self.stdout.write(f"Recall: {metrics['recall']:.4f}")
                self.stdout.write(f"F1 Score: {metrics['f1_score']:.4f}")
                
                # Print ranking metrics
                ranking_metrics = metrics['ranking_metrics']
                self.stdout.write("\nRanking Metrics:")
                self.stdout.write(f"Precision@{options['components']}: {ranking_metrics['precision@k']:.4f}")
                self.stdout.write(f"Recall@{options['components']}: {ranking_metrics['recall@k']:.4f}")
                self.stdout.write(f"NDCG@{options['components']}: {ranking_metrics['ndcg@k']:.4f}")
                self.stdout.write(f"MAP@{options['components']}: {ranking_metrics['map@k']:.4f}")
                self.stdout.write(f"Hit Rate@{options['components']}: {ranking_metrics['hit_rate@k']:.4f}")
                
                # Confirm plot generation
                plot_path = f'instacart/static/images/explained_variance_cluster_{cluster_id}.png'
                if os.path.exists(plot_path):
                    self.stdout.write(f"\nExplained variance plot saved to: {plot_path}")
                else:
                    self.stdout.write(self.style.WARNING(f"Warning: Plot not generated for cluster {cluster_id}"))
        else:
            self.stdout.write(
                self.style.ERROR('Failed to train SVD cluster models')
            ) 