from django.core.management.base import BaseCommand
import pandas as pd
from instacart.models import Aisle, Department, Product, Order, OrderProduct

class Command(BaseCommand):
    help = 'Load data from CSV files'

    def handle(self, *args, **kwargs):
        # Load Aisles
        aisles_df = pd.read_csv('instacart/static/csv/aisles.csv')
        for _, row in aisles_df.iterrows():
            Aisle.objects.create(aisle_id=row['aisle_id'], aisle=row['aisle'])

        # Load Departments
        departments_df = pd.read_csv('instacart/static/csv/departments.csv')
        for _, row in departments_df.iterrows():
            Department.objects.create(department_id=row['department_id'], department=row['department'])

        # Load Products
        products_df = pd.read_csv('instacart/static/csv/products.csv')
        for _, row in products_df.iterrows():
            Product.objects.create(
                product_id=row['product_id'],
                product_name=row['product_name'],
                aisle_id=row['aisle_id'],
                department_id=row['department_id']
            )

        # Load Orders
        orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
        for _, row in orders_df.iterrows():
            Order.objects.create(
                order_id=row['order_id'],
                user_id=row['user_id'],
                eval_set=row['eval_set'],
                order_number=row['order_number'],
                order_dow=row['order_dow'],
                order_hour_of_day=row['order_hour_of_day'],
                days_since_prior_order=row['days_since_prior_order']
            )

        # Load Order Products
        order_products_df = pd.read_csv('instacart/static/csv/merged_data.csv')
        for _, row in order_products_df.iterrows():
            OrderProduct.objects.create(
                order_id=row['order_id'],
                product_id=row['product_id'],
                add_to_cart_order=row['add_to_cart_order'],
                reordered=row['reordered']
            ) 