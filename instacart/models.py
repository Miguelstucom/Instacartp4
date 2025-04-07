from django.db import models
from django.contrib.auth.models import User

class Aisle(models.Model):
    aisle_id = models.IntegerField(primary_key=True)
    aisle = models.CharField(max_length=100)

    def __str__(self):
        return self.aisle

class Department(models.Model):
    department_id = models.IntegerField(primary_key=True)
    department = models.CharField(max_length=100)

    def __str__(self):
        return self.department

class Product(models.Model):
    product_id = models.IntegerField(primary_key=True)
    product_name = models.CharField(max_length=200)
    aisle = models.ForeignKey(Aisle, on_delete=models.CASCADE)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)

    def __str__(self):
        return self.product_name

class Order(models.Model):
    order_id = models.IntegerField(primary_key=True)
    user_id = models.IntegerField()
    eval_set = models.CharField(max_length=20)
    order_number = models.IntegerField()
    order_dow = models.IntegerField()
    order_hour_of_day = models.IntegerField()
    days_since_prior_order = models.FloatField(null=True)

class OrderProduct(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    add_to_cart_order = models.IntegerField()
    reordered = models.IntegerField()

class UserSession(models.Model):
        user_id = models.IntegerField(unique=True)
        last_login = models.DateTimeField(auto_now=True)

        def str(self):
            return f"User {self.user_id}"

class Cart(models.Model):
        user_id = models.IntegerField()
        created_at = models.DateTimeField(auto_now_add=True)

        class Meta:
            indexes = [
                models.Index(fields=['user_id'])
            ]

class CartItem(models.Model):
        cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
        product_id = models.IntegerField()
        product_name = models.CharField(max_length=255)
        quantity = models.IntegerField(default=1)
        added_at = models.DateTimeField(auto_now_add=True)
