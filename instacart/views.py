from django.shortcuts import render, redirect

from django.utils import timezone
from django.contrib import messages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Order, OrderProduct, Product, Aisle, UserSession, Cart
import pandas as pd
from .ml_utils import MarketBasketModel, SVDRecommender

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
        mba_model = MarketBasketModel.load_model()
        svd_model = SVDRecommender.load_model()
        
        if mba_model is None and svd_model is None:
            messages.warning(request, "Recommendation models not available. Please train the models first.")
            return render(request, 'home.html', {'user_id': user_id})
        
        # Get user's previous purchases
        products_df, merged_df, orders_df = load_data()
        user_aisles = merged_df.merge(
            orders_df[orders_df['user_id'] == user_id][['order_id']],
            on='order_id'
        ).merge(
            products_df[['product_id', 'aisle_id']],
            on='product_id'
        )['aisle_id'].unique().tolist()
        
        # Initialize context
        context = {
            'user_id': user_id,
            'other_aisles': Aisle.objects.all(),
            'cluster_info': cluster_info,
            'user_metrics': {
                'total_orders': int(user_info['Frequency']),
                'avg_order_value': float(user_info['Monetary'] / user_info['Frequency']),
                'days_since_last': int(user_info['Recency'])
            }
        }
        
        # Get MBA recommendations if available
        if mba_model is not None:
            # Get recommendations with cluster weights
            mba_recommendations = mba_model.get_recommendations(user_id=user_id, user_aisles=user_aisles)
            
            # Sort by the cluster-weighted score
            mba_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            context.update({
                'mba_recommendations': mba_recommendations[:5],  # Get top 5 recommendations
                'mba_metrics': mba_model.stored_metrics,
                'recommended_aisles': mba_recommendations[:5]
            })
        
        # Get SVD recommendations if available
        if svd_model is not None:
            svd_recommendations = svd_model.get_recommendations(user_id)
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
        for rec in combined_list:
            aisle_products = []
            if mba_model is not None:
                aisle_products = mba_model.get_aisle_products(user_id, rec['aisle_id'])
            rec['recommended_products'] = aisle_products
        
        # Add to context
        context['combined_recommendations'] = combined_list
        
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
        products_df, _, _ = load_data()
        
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
        
        # Get top 10 most similar products
        top_indices = similarity_scores[0].argsort()[-10:][::-1]
        results = products_df.iloc[top_indices][['product_id', 'product_name', 'aisle', 'department']]
        
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





