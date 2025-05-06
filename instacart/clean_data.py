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
optimal_k = 4  # Puedes cambiarlo según el gráfico
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

# Aisle Clustering
print("\nPerforming Aisle Clustering...")

# Crear matriz binaria de interacción pasillo-pedido
aisle_order_matrix = filtered_merged_data.merge(
    filtered_orders[['order_id']],
    on='order_id'
).merge(
    pd.read_csv('instacart/static/csv/products.csv')[['product_id', 'aisle_id']],
    on='product_id'
).drop_duplicates(subset=['order_id', 'aisle_id'])  # una sola vez por pedido

# Matriz de ocurrencias (filas: orders, columnas: aisles)
order_aisle_crosstab = aisle_order_matrix.groupby(['order_id', 'aisle_id']).size().unstack(fill_value=0)
order_aisle_crosstab = (order_aisle_crosstab > 0).astype(int)

# Transponer para tener: filas = aisles, columnas = pedidos
aisle_matrix = order_aisle_crosstab.T

# Escalar
scaler = StandardScaler()
aisle_matrix_scaled = scaler.fit_transform(aisle_matrix)

# Encontrar la k óptima
wcss_aisles = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(aisle_matrix_scaled)
    wcss_aisles.append(kmeans.inertia_)

# Gráfico del codo
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), wcss_aisles, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.title('Método del Codo para elegir k (Pasillos por co-ocurrencia)')
plt.show()

# Aplicar K-Means con k óptimo
optimal_k_aisles = 6  # o el que decidas tras ver el gráfico
kmeans_aisles = KMeans(n_clusters=optimal_k_aisles, random_state=42, n_init=10)
aisle_clusters = kmeans_aisles.fit_predict(aisle_matrix_scaled)

# Crear DataFrame con los resultados
aisle_info = pd.DataFrame({
    'aisle_id': aisle_matrix.index,
    'Cluster': aisle_clusters
})

# Añadir nombres de pasillos
aisles_df = pd.read_csv('instacart/static/csv/aisles.csv')
aisle_info = aisle_info.merge(aisles_df, on='aisle_id')

# Guardar los clusters
aisle_info.to_csv('instacart/static/csv/aisle_clusters.csv', index=False)

# Estadísticas por cluster
print("\nAisle Cluster Statistics:")
aisle_cluster_stats = aisle_info.groupby("Cluster").agg({
    'aisle_id': 'count'
}).rename(columns={'aisle_id': 'Number of Aisles'})
print(aisle_cluster_stats)

# Descripciones opcionales
print("\nAisle Cluster Descriptions:")
aisle_cluster_descriptions = {
    0: "Aisles with strong co-purchase links",
    1: "Independent or niche aisles",
    2: "Frequently co-occurring core aisles",
    3: "Complementary shopping zones",
    # Agrega descripciones según lo que observes
}

for cluster, description in aisle_cluster_descriptions.items():
    print(f"Cluster {cluster}: {description}")
    cluster_aisles = aisle_info[aisle_info['Cluster'] == cluster]['aisle'].tolist()
    print(f"Aisles in this cluster:")
    print(cluster_aisles)
    print()

# Visualización con t-SNE
print("\nVisualizando Clusters de Pasillos con t-SNE...")

tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
aisle_tsne = tsne.fit_transform(aisle_matrix_scaled)

# Preparar para graficar
aisle_viz = pd.DataFrame(aisle_tsne, columns=['Dim1', 'Dim2'])
aisle_viz['Cluster'] = aisle_clusters
aisle_viz['aisle'] = aisle_info['aisle'].values

plt.figure(figsize=(12, 8))
sns.scatterplot(data=aisle_viz, x='Dim1', y='Dim2', hue='Cluster', palette='tab10', legend='full', alpha=0.7)
plt.title('Clusters de Pasillos (t-SNE) por co-ocurrencia en pedidos')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# Cargar los datos necesarios
products_df = pd.read_csv('instacart/static/csv/products.csv')[['product_id', 'aisle_id']]

# Relacionar user_id con productos
user_product_data = filtered_merged_data.merge(filtered_orders[['order_id', 'user_id']], on='order_id')
user_product_data = user_product_data.merge(products_df, on='product_id')

# Añadir el cluster de cada usuario
user_product_data = user_product_data.merge(rfm[['user_id', 'Cluster']], on='user_id')

# Contar cuántos aisle_id únicos hay por cluster
aisles_por_cluster = user_product_data.groupby('Cluster')['aisle_id'].nunique()

# Mostrar resultados
print("Cantidad de aisles diferentes comprados por cada user cluster:\n")
print(aisles_por_cluster)
