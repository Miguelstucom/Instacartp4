from django.core.management.base import BaseCommand
from instacart.ml_utils import SVDClusterRecommender

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
        
        # Initialize and train the model
        model = SVDClusterRecommender(n_components=options['components'])
        success = model.train_model(test_size=options['test_size'])
        
        if success:
            # Save the trained model
            model.save_model()
            self.stdout.write(
                self.style.SUCCESS('Successfully trained and saved SVD cluster models')
            )
        else:
            self.stdout.write(
                self.style.ERROR('Failed to train SVD cluster models')
            ) 