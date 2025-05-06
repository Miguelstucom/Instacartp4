from django.core.management.base import BaseCommand
from instacart.ml_utils import ClusterProductBasketModel

class Command(BaseCommand):
    help = 'Train the cluster-based product basket analysis model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--min-support',
            type=float,
            default=0.01,
            help='Minimum support for frequent itemsets'
        )
        parser.add_argument(
            '--min-lift',
            type=float,
            default=1.5,
            help='Minimum lift for association rules'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Test set size for model evaluation'
        )

    def handle(self, *args, **options):
        self.stdout.write('Training cluster-based product basket model...')
        
        # Create and train the model
        model = ClusterProductBasketModel()
        success = model.train_model(
            min_support=options['min_support'],
            min_lift=options['min_lift'],
            test_size=options['test_size']
        )
        
        if success:
            # Save the model
            if model.save_model():
                self.stdout.write(self.style.SUCCESS('Model trained and saved successfully!'))
            else:
                self.stdout.write(self.style.ERROR('Failed to save the model'))
        else:
            self.stdout.write(self.style.ERROR('Failed to train the model')) 