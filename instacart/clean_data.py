import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Cargar los datos
orders = pd.read_csv('instacart/static/csv/orders.csv')
merged_data = pd.read_csv('instacart/static/csv/merged_data.csv')

# Calculate 2% of total orders
n_orders = int(len(orders) * 1)

# Take the first 2% of order_ids
orders_to_keep = orders['order_id'].head(n_orders)

# Filter the orders and merged data
filtered_orders = orders[orders['order_id'].isin(orders_to_keep)]
filtered_merged_data = merged_data[merged_data['order_id'].isin(orders_to_keep)]

# Manejo de valores nulos en days_since_prior_order
def clean_days_since_prior_order(x):
    if x.isnull().all():
        return x.fillna(0)
    elif 0 in x.values:
        return x.dropna()
    return x.fillna(0)

filtered_orders['days_since_prior_order'] = filtered_orders.groupby('user_id')['days_since_prior_order'].transform(clean_days_since_prior_order)

# Recency: Última compra de cada usuario
recency = filtered_orders.groupby("user_id")["days_since_prior_order"].max().reset_index()
recency.columns = ["user_id", "Recency"]

# Frequency: Número total de pedidos por usuario
frequency = filtered_orders.groupby("user_id")["order_number"].max().reset_index()
frequency.columns = ["user_id", "Frequency"]

# Monetary: Total de productos comprados por usuario
monetary = filtered_merged_data.groupby("order_id")["product_id"].count().reset_index()
monetary = monetary.merge(filtered_orders[['order_id', 'user_id']], on='order_id')
monetary = monetary.groupby("user_id")["product_id"].sum().reset_index()
monetary.columns = ["user_id", "Monetary"]

# Unir todas las métricas
rfm = recency.merge(frequency, on="user_id").merge(monetary, on="user_id")

# Normalizar los datos
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Encontrar la k óptima con el método del codo
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Graficar el codo
plt.plot(range(2, 10), wcss, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.title('Método del Codo para elegir k')
plt.show()

# Aplicar K-Means con el k óptimo
optimal_k = 5  # Puedes cambiarlo según el gráfico
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Mostrar la media de cada cluster
print(rfm.groupby("Cluster").mean())

plt.figure(figsize=(8, 5))
plt.scatter(rfm_scaled[:, 0], rfm_scaled[:, 1], c=rfm["Cluster"], cmap='viridis', alpha=0.6)
plt.xlabel("Recency (Normalizado)")
plt.ylabel("Frequency (Normalizado)")
plt.title("Clusters de Usuarios")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()



# After creating rfm DataFrame and assigning clusters, save user information
user_clusters = rfm.copy()
user_clusters.reset_index(inplace=True)  # Make user_id a column instead of index
user_clusters.to_csv('instacart/static/csv/user_clusters.csv', index=False)

print("\nCluster Descriptions:")
cluster_descriptions = {
    0: "Super Frequent Customers - Recent small purchases, very high frequency",
    1: "Inactive/At Risk Customers - Not recent, low frequency and spending",
    2: "VIP/Big Spenders - Medium recency, high frequency and spending",
    3: "Growing Customers - Recent purchases, moderate frequency and spending",
    4: "New/Sporadic Customers - Very recent but low frequency and spending"
}

# Print cluster statistics
print("\nCluster Statistics:")
cluster_stats = rfm.groupby("Cluster").agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'user_id': 'count'
}).round(2)

cluster_stats['Description'] = cluster_stats.index.map(cluster_descriptions)
print(cluster_stats)

# Save cluster statistics
cluster_stats.to_csv('instacart/static/csv/cluster_statistics.csv')

filtered_orders.sort_values('order_id').to_csv('instacart/static/csv/clean_orders.csv', index=False)
filtered_merged_data.sort_values('order_id').to_csv('instacart/static/csv/clean_merged_data.csv', index=False)
