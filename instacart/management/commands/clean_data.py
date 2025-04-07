from django.core.management.base import BaseCommand
from instacart.data_utils import clean_all_data

class Command(BaseCommand):
    help = 'Cleans the CSV data files'

    def handle(self, *args, **options):
        self.stdout.write('Iniciando proceso de limpieza de datos...')
        
        try:
            clean_all_data()
            self.stdout.write(self.style.SUCCESS('Datos limpiados exitosamente'))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error durante la limpieza de datos: {str(e)}')
            ) 