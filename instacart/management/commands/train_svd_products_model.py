from django.core.management.base import BaseCommand
from instacart.ml_utils import SVDProductRecommender

class Command(BaseCommand):
    help = 'Trains the SVD product recommendation model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--components',
            type=int,
            default=80,
            help='Number of components for SVD'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Proportion of data to use for testing'
        )

    def handle(self, *args, **options):
        self.stdout.write('Starting SVD product model training...')
        
        model = SVDProductRecommender(n_components=options['components'])
        success = model.train_model(test_size=options['test_size'])
        
        if success:
            model.save_model()
            self.stdout.write(self.style.SUCCESS('SVD product model trained and saved successfully'))
        else:
            self.stdout.write(self.style.ERROR('Error during SVD product model training'))
