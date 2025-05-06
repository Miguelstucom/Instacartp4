from django.shortcuts import render, redirect

from django.utils import timezone
from django.contrib import messages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Order, OrderProduct, Product, Aisle, UserSession, Cart
import pandas as pd
from .ml_utils import MarketBasketModel, SVDRecommender, SVDClusterRecommender, ClusterProductBasketModel

# Create your views here.

def load_data():
    products_df = pd.read_csv('instacart/static/csv/products.csv')
    merged_df = pd.read_csv('instacart/static/csv/merged_data.csv')
    orders_df = pd.read_csv('instacart/static/csv/orders_cleaned.csv')
    aisles_df = pd.read_csv('instacart/static/csv/aisles.csv')
    departments_df = pd.read_csv('instacart/static/csv/departments.csv')

    products_df = products_df.merge(aisles_df, on='aisle_id')
    products_df = products_df.merge(departments_df, on='department_id')

    return products_df, merged_df, orders_df

def login(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        
        try:
            # Validar que el user_id sea un número
            user_id = int(user_id)
            
            products_df, merged_df, orders_df = load_data()
            
            if user_id in orders_df['user_id'].unique():
                # Obtener las compras previas del usuario
                user_orders = merged_df.merge(
                    orders_df[orders_df['user_id'] == user_id][['order_id', 'user_id']],
                    on='order_id'
                )
                user_products = user_orders.merge(
                    products_df[['product_id', 'product_name']],
                    on='product_id'
                )[['product_id', 'product_name']].drop_duplicates()

                # Guardar información en la sesión
                request.session['user_id'] = user_id
                request.session['last_login'] = str(timezone.now())

                # Actualizar o crear sesión de usuario
                UserSession.objects.update_or_create(
                    user_id=user_id,
                    defaults={'user_id': user_id}
                )

                # Crear un nuevo carrito para el usuario
                Cart.objects.create(user_id=user_id)

                return redirect('home')
            else:
                messages.error(request, 'Usuario no encontrado en nuestra base de datos.')
        except ValueError:
            messages.error(request, 'Por favor ingrese un ID de usuario válido (número entero).')
        except Exception as e:
            messages.error(request, f'Error al procesar la solicitud: {str(e)}')

    return render(request, 'login.html')

def home(request):
    if 'user_id' not in request.session:
        messages.error(request, 'Please login first')
        return redirect('login')
    
    user_id = request.session['user_id']
    
    try:
        # Load user cluster information
        user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
        user_info = user_clusters[user_clusters['user_id'] == user_id].iloc[0]
        
        # Get cluster description
        cluster_descriptions = {
            0: "Super Frequent Customer",
            1: "Inactive/At Risk Customer",
            2: "VIP/Big Spender",
            3: "Growing Customer",
            4: "New/Sporadic Customer"
        }
        
        cluster_info = {
            'cluster': int(user_info['Cluster']),
            'description': cluster_descriptions[user_info['Cluster']],
            'recency': user_info['Recency'],
            'frequency': user_info['Frequency'],
            'monetary': user_info['Monetary']
        }
        
        # Load both models
        print("carga el mba")
        mba_model = MarketBasketModel.load_model()
        print("carga el svd")
        svd_model = SVDRecommender.load_model()
        print("carga el cluster svd")
        cluster_svd_model = SVDClusterRecommender.load_model()
        print("cargados")
        
        if mba_model is None and svd_model is None and cluster_svd_model is None:
            messages.warning(request, "Recommendation models not available. Please train the models first.")
            return render(request, 'home.html', {'user_id': user_id})
        print("tarda 2")
        
        # Get user's previous purchases
        products_df, merged_df, orders_df = load_data()
        user_aisles = merged_df.merge(
            orders_df[orders_df['user_id'] == user_id][['order_id']],
            on='order_id'
        ).merge(
            products_df[['product_id', 'aisle_id']],
            on='product_id'
        )['aisle_id'].unique().tolist()

        # Get detailed order information for the user
        user_orders = orders_df[orders_df['user_id'] == user_id].sort_values('order_number')
        order_details = []
        triggered_rules = set()
        
        for _, order in user_orders.iterrows():
            # Get products and aisles for this order
            order_products = merged_df[merged_df['order_id'] == order['order_id']]
            order_products = order_products.merge(
                products_df[['product_id', 'product_name', 'aisle_id']],
                on='product_id'
            )
            
            # Get unique aisles in this order
            order_aisles = order_products['aisle_id'].unique().tolist()
            
            # Check which MBA rules were triggered
            triggered_rules_in_order = []
            if mba_model is not None and mba_model.rules is not None:
                for _, rule in mba_model.rules.iterrows():
                    antecedents = rule['antecedents']
                    consequents = rule['consequents']
                    
                    # Check if antecedents are in the order
                    if all(aid in order_aisles for aid in antecedents):
                        # Check if consequents are also in the order
                        if any(cid in order_aisles for cid in consequents):
                            triggered_rules_in_order.append({
                                'antecedents': [mba_model.aisle_names.get(aid, f"Aisle {aid}") for aid in antecedents],
                                'consequents': [mba_model.aisle_names.get(cid, f"Aisle {cid}") for cid in consequents],
                                'confidence': float(rule['confidence']),
                                'lift': float(rule['lift']),
                                'support': float(rule['support'])
                            })
                            triggered_rules.add((tuple(antecedents), tuple(consequents)))
            
            order_details.append({
                'order_id': order['order_id'],
                'order_number': order['order_number'],
                'order_dow': order['order_dow'],
                'order_hour': order['order_hour_of_day'],
                'aisles': [mba_model.aisle_names.get(aid, f"Aisle {aid}") for aid in order_aisles],
                'triggered_rules': triggered_rules_in_order
            })
        
        # Get summary of triggered rules
        rule_summary = {
            'total_rules_triggered': len(triggered_rules),
            'unique_rules': [
                {
                    'antecedents': [mba_model.aisle_names.get(aid, f"Aisle {aid}") for aid in antecedents],
                    'consequents': [mba_model.aisle_names.get(cid, f"Aisle {cid}") for cid in consequents]
                }
                for antecedents, consequents in triggered_rules
            ]
        }

        print("tarda en datos")
        
        # Initialize context
        context = {
            'user_id': user_id,
            'other_aisles': Aisle.objects.all(),
            'cluster_info': cluster_info,
            'user_metrics': {
                'total_orders': int(user_info['Frequency']),
                'avg_order_value': float(user_info['Monetary'] / user_info['Frequency']),
                'days_since_last': int(user_info['Recency'])
            },
            'order_details': order_details,
            'rule_summary': rule_summary
        }
        
        # Get MBA recommendations if available
        if mba_model is not None:
            # Get recommendations with cluster weights
            mba_recommendations = mba_model.get_recommendations(user_id=user_id, user_aisles=user_aisles, n_recommendations=5)
            
            # Sort by the cluster-weighted score
            mba_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Get user's cluster
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['Cluster'].iloc[0]
            cluster_users = user_clusters[user_clusters['Cluster'] == user_cluster]['user_id'].tolist()
            
            # Add top 3 products for each aisle
            for rec in mba_recommendations:
                aisle_id = rec['aisle_id']
                # Get top 3 products in this aisle for the user's cluster
                top_products = (merged_df
                    .merge(orders_df[orders_df['user_id'].isin(cluster_users)][['order_id']], on='order_id')
                    .merge(products_df[products_df['aisle_id'] == aisle_id][['product_id', 'product_name']], on='product_id')
                    .groupby(['product_id', 'product_name'])
                    .size()
                    .reset_index(name='cluster_count')
                    .sort_values('cluster_count', ascending=False)
                    .head(3)
                )
                rec['top_products'] = top_products.to_dict('records')
            
            context.update({
                'mba_recommendations': mba_recommendations,
                'mba_metrics': mba_model.stored_metrics,
                'recommended_aisles': mba_recommendations
            })
        
        # Get SVD recommendations if available
        if svd_model is not None:
            svd_recommendations = svd_model.get_recommendations(user_id)
            
            # Add top 3 products for each aisle
            for rec in svd_recommendations:
                aisle_id = rec['aisle_id']
                # Get top 3 products in this aisle for the user's cluster
                top_products = (merged_df
                    .merge(orders_df[orders_df['user_id'].isin(cluster_users)][['order_id']], on='order_id')
                    .merge(products_df[products_df['aisle_id'] == aisle_id][['product_id', 'product_name']], on='product_id')
                    .groupby(['product_id', 'product_name'])
                    .size()
                    .reset_index(name='cluster_count')
                    .sort_values('cluster_count', ascending=False)
                    .head(3)
                )
                rec['top_products'] = top_products.to_dict('records')
            
            context.update({
                'svd_recommendations': svd_recommendations,
                'svd_metrics': svd_model.stored_metrics
            })
        
        # Update other_aisles to exclude recommended aisles
        if 'mba_recommendations' in context:
            recommended_aisle_ids = [rec['aisle_id'] for rec in context['mba_recommendations']]
            context['other_aisles'] = [aisle for aisle in context['other_aisles'] 
                                     if aisle.aisle_id not in recommended_aisle_ids]
        
        # Prepare combined recommendations
        combined_recommendations = {}
        
        # Add MBA recommendations to combined dict
        if mba_model is not None and 'mba_recommendations' in context:
            for rec in context['mba_recommendations']:
                aisle_id = rec['aisle_id']
                if aisle_id not in combined_recommendations:
                    combined_recommendations[aisle_id] = {
                        'aisle_id': aisle_id,
                        'aisle_name': rec['aisle_name'],
                        'sources': ['MBA'],
                        'mba_score': rec['score'],
                        'mba_confidence': rec['confidence'],
                        'mba_lift': rec['lift'],
                        'svd_score': None
                    }
        
        # Add SVD recommendations to combined dict
        if svd_model is not None and 'svd_recommendations' in context:
            for rec in context['svd_recommendations']:
                aisle_id = rec['aisle_id']
                if aisle_id not in combined_recommendations:
                    combined_recommendations[aisle_id] = {
                        'aisle_id': aisle_id,
                        'aisle_name': rec['aisle_name'],
                        'sources': ['SVD'],
                        'svd_score': rec['similarity_score'],
                        'mba_score': None,
                        'mba_confidence': None,
                        'mba_lift': None
                    }
                else:
                    # Aisle already exists from MBA, add SVD info
                    combined_recommendations[aisle_id]['sources'].append('SVD')
                    combined_recommendations[aisle_id]['svd_score'] = rec['similarity_score']
        
        # Convert to list and sort by combined score
        combined_list = list(combined_recommendations.values())
        for rec in combined_list:
            # Calculate combined score
            mba_score = rec['mba_score'] if rec['mba_score'] is not None else 0
            svd_score = rec['svd_score'] if rec['svd_score'] is not None else 0
            rec['combined_score'] = (mba_score + svd_score) / len(rec['sources'])
        
        # Sort by combined score
        combined_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Get personalized products for each aisle
        
        # Get cluster-specific product recommendations
        cluster_product_recommendations = []
        if cluster_svd_model is not None:
            cluster_product_recommendations = cluster_svd_model.get_recommendations(user_id, n_recommendations=9)
        
        # Get top 10 most sold products for the user's cluster
        top_sold_products = []
        try:
            # Load data
            products_df, merged_df, orders_df = load_data()
            user_clusters = pd.read_csv('instacart/static/csv/user_clusters.csv')
            
            # Get user's cluster
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['Cluster'].iloc[0]
            
            # Get users in the same cluster
            cluster_users = user_clusters[user_clusters['Cluster'] == user_cluster]['user_id'].tolist()
            
            # Get top 10 most sold products in the cluster
            top_products = (merged_df
                .merge(orders_df[orders_df['user_id'].isin(cluster_users)][['order_id']], on='order_id')
                .groupby('product_id')
                .size()
                .reset_index(name='count')
                .sort_values('count', ascending=False)
                .head(10)
            )
            
            # Add product names
            top_products = top_products.merge(
                products_df[['product_id', 'product_name']],
                on='product_id'
            )
            
            # Convert to list of dictionaries
            top_sold_products = top_products.to_dict('records')
            
        except Exception as e:
            print(f"Error getting top sold products: {str(e)}")
        
        # Add to context
        context['combined_recommendations'] = combined_list
        context['cluster_product_recommendations'] = cluster_product_recommendations
        context['top_sold_products'] = top_sold_products
        return render(request, 'home.html', context)
        
    except Exception as e:
        print(f"Error in home view: {str(e)}")
        messages.error(request, f'Error loading home page: {str(e)}')
        return redirect('login')

def search(request):
    """Handle search functionality"""
    query = request.GET.get('q', '')
    if not query:
        return redirect('home')
        
    try:
        # Load product data
        products_df, merged_df, orders_df = load_data()
        
        # Create a text field combining product name, aisle, and department
        products_df['search_text'] = products_df['product_name'] + ' ' + \
                                   products_df['aisle'] + ' ' + \
                                   products_df['department']
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(products_df['search_text'])
        
        # Transform search query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
        
        # Get all products with their NLP scores
        results = products_df[['product_id', 'product_name', 'aisle', 'department']].copy()
        results['nlp_score'] = similarity_scores[0]
        
        # Load SVD cluster model for recommendations
        svd_model = SVDClusterRecommender.load_model()
        
        # Get user's previously bought products if logged in
        previously_bought = set()
        if 'user_id' in request.session:
            user_id = request.session['user_id']
            user_products = merged_df.merge(
                orders_df[orders_df['user_id'] == user_id][['order_id']],
                on='order_id'
            )['product_id'].unique()
            previously_bought = set(user_products)
            
            # Get SVD recommendations for all products
            if svd_model is not None:
                # Get all recommendations at once
                all_recommendations = svd_model.get_recommendations(user_id, n_recommendations=1000)
                
                # Create a dictionary of product_id to score
                svd_scores = {rec['product_id']: rec['similarity_score'] for rec in all_recommendations}
                
                # Add SVD scores to results, defaulting to 0 if not found
                results['svd_score'] = results['product_id'].map(lambda x: svd_scores.get(x, 0))
            else:
                results['svd_score'] = 0
        else:
            results['svd_score'] = 0
        
        # Add previously bought flag
        results['previously_bought'] = results['product_id'].isin(previously_bought)
        
        # Calculate combined score (70% NLP, 30% SVD)
        results['combined_score'] = (0.7 * results['nlp_score']) + (0.3 * results['svd_score'])
        
        # Sort by combined score and get top 10
        results = results.sort_values('combined_score', ascending=False).head(10)
        
        # Group results by aisle
        aisle_results = results.groupby('aisle').agg({
            'product_id': 'count',
            'product_name': list,
            'department': 'first'
        }).reset_index()
        
        context = {
            'query': query,
            'results': results.to_dict('records'),
            'aisle_results': aisle_results.to_dict('records'),
            'total_results': len(results)
        }
        
        return render(request, 'search.html', context)
        
    except Exception as e:
        messages.error(request, f'Error during search: {str(e)}')
        return redirect('home')

def logout(request):
    """Handle user logout"""
    if 'user_id' in request.session:
        del request.session['user_id']
    messages.success(request, 'You have been logged out successfully')
    return redirect('login')

def product_detail(request, product_id):
    try:
        # Load product data
        products_df, merged_df, orders_df = load_data()
        
        # Get product details
        product = products_df[products_df['product_id'] == product_id].iloc[0]
        
        # Load product basket model
        product_basket_model = ClusterProductBasketModel.load_model()
        
        # Get frequently bought together recommendations
        frequently_bought_together = []
        if product_basket_model is not None and 'user_id' in request.session:
            frequently_bought_together = product_basket_model.get_recommendations(
                product_id=product_id,
                user_id=request.session['user_id'],
                n_recommendations=5
            )
        
        context = {
            'product': product,
            'frequently_bought_together': frequently_bought_together
        }
        
        return render(request, 'product_detail.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading product details: {str(e)}')
        return redirect('home')





