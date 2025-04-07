from django.core.management.base import BaseCommand
from instacart.ml_utils import MarketBasketModel

class Command(BaseCommand):
    help = 'Trains the market basket analysis model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--min-support',
            type=float,
            default=0.05,
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

    def handle(self, *args, **options):
        self.stdout.write('Iniciando entrenamiento del modelo...')
        
        model = MarketBasketModel()
        success = model.train_model(
            min_support=options['min_support'],
            min_lift=options['min_lift'],
            test_size=options['test_size'],
            random_state=options['random_state']
        )
        
        if success:
            model.save_model()
            self.stdout.write(self.style.SUCCESS('Modelo entrenado y guardado exitosamente'))
            
            # Calculate and display model metrics
            metrics = model.calculate_metrics()
            if metrics:
                self.stdout.write('\nModel Performance Metrics:')
                self.stdout.write('-' * 30)
                self.stdout.write(f"Accuracy:   {metrics['accuracy']:.3f}")
                self.stdout.write(f"Precision:  {metrics['precision']:.3f}")
                self.stdout.write(f"Recall:     {metrics['recall']:.3f}")
                self.stdout.write(f"F1 Score:   {metrics['f1_score']:.3f}")
                self.stdout.write('\nDetailed Statistics:')
                self.stdout.write('-' * 30)
                self.stdout.write(f"True Positives:  {metrics['true_positives']}")
                self.stdout.write(f"False Positives: {metrics['false_positives']}")
                self.stdout.write(f"True Negatives:  {metrics['true_negatives']}")
                self.stdout.write(f"False Negatives: {metrics['false_negatives']}")
                self.stdout.write('\nRule Statistics:')
                self.stdout.write('-' * 30)
                self.stdout.write(f"Total Rules:      {metrics['total_rules']}")
                self.stdout.write(f"Avg Confidence:   {metrics['avg_confidence']:.3f}")
                self.stdout.write(f"Avg Lift:         {metrics['avg_lift']:.3f}")
                self.stdout.write(f"Avg Support:      {metrics['avg_support']:.3f}")
        else:
            self.stdout.write(self.style.ERROR('Error durante el entrenamiento del modelo')) 