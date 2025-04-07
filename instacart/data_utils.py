import pandas as pd
import numpy as np

def clean_days_since_prior_order(x):
    """
    Limpia los valores nulos en days_since_prior_order
    - Si todos son nulos, rellena con 0
    - Si hay valores 0, elimina los nulos
    - En otro caso, rellena con 0
    """
    if x.isnull().all():
        return x.fillna(0)
    elif 0 in x.values:
        return x.dropna()
    return x.fillna(0)

def clean_orders_data(file_path='instacart/static/csv/orders_cleaned.csv', output_path='instacart/static/csv/orders_cleaned.csv'):
    """
    Limpia el archivo de órdenes y guarda una versión limpia
    """
    try:
        # Cargar datos
        print("Cargando archivo de órdenes...")
        orders_df = pd.read_csv(file_path)
        
        # Guardar número de filas original
        original_rows = len(orders_df)
        
        # Aplicar limpieza a days_since_prior_order
        print("Limpiando datos...")
        orders_df['days_since_prior_order'] = clean_days_since_prior_order(orders_df['days_since_prior_order'])
        
        # Eliminar filas donde aún haya nulos (si es necesario)
        orders_df = orders_df.dropna(subset=['days_since_prior_order'])
        
        # Guardar archivo limpio
        print("Guardando archivo limpio...")
        orders_df.to_csv(output_path, index=False)
        
        # Reportar resultados
        final_rows = len(orders_df)
        removed_rows = original_rows - final_rows
        
        print(f"Proceso completado:")
        print(f"- Filas originales: {original_rows}")
        print(f"- Filas eliminadas: {removed_rows}")
        print(f"- Filas finales: {final_rows}")
        print(f"Archivo guardado en: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error durante la limpieza de datos: {str(e)}")
        return False

def clean_all_data():
    """
    Función principal para limpiar todos los datasets necesarios
    """
    print("Iniciando limpieza de datos...")
    
    # Limpiar orders.csv
    if clean_orders_data():
        print("Limpieza de orders.csv completada exitosamente")
    else:
        print("Error en la limpieza de orders.csv")
    
    # Aquí puedes agregar más funciones de limpieza para otros archivos si es necesario
    
    print("Proceso de limpieza finalizado") 