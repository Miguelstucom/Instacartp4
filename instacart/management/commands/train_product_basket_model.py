from django.core.management.base import BaseCommand
from instacart.ml_utils import ProductBasketModel

class Command(BaseCommand):
    help = 'Trains the product-based basket analysis model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--min-support',
            type=float,
            default=0.01,
            help='Minimum support threshold for Apriori algorithm'
        )
        parser.add_argument(
            '--min-lift',
            type=float,
            default=1.0,
            help='Minimum lift threshold for association rules'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Proportion of data to use for testing'
        )
        parser.add_argument(
            '--random-state',
            type=int,
            default=42,
            help='Random seed for reproducibility'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10000,
            help='Number of orders to process in each batch'
        )

    def handle(self, *args, **options):
        self.stdout.write('Starting product basket model training...')
        
        model = ProductBasketModel(batch_size=options['batch_size'])
        success = model.train_model(
            min_support=options['min_support'],
            min_lift=options['min_lift'],
            test_size=options['test_size'],
            random_state=options['random_state']
        )
        
        if success:
            model.save_model()
            self.stdout.write(self.style.SUCCESS('Model trained and saved successfully'))
            
            # Display model metrics
            metrics = model.stored_metrics
            if metrics:
                self.stdout.write('\nModel Performance Metrics:')
                self.stdout.write('-' * 30)
                self.stdout.write(f"Accuracy:   {metrics['accuracy']:.3f}")
                self.stdout.write(f"Precision:  {metrics['precision']:.3f}")
                self.stdout.write(f"Recall:     {metrics['recall']:.3f}")
                self.stdout.write(f"F1 Score:   {metrics['f1_score']:.3f}")
        else:
            self.stdout.write(self.style.ERROR('Model training failed')) 