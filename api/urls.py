from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    # User submission endpoints
    path('submissions/', views.SubmissionBatchCreateView.as_view(), name='create-submission'),
    path('submissions/list/', views.SubmissionBatchListView.as_view(), name='list-submissions'),
    path('submissions/<int:pk>/', views.SubmissionBatchDetailView.as_view(), name='submission-detail'),
    
    # User product endpoints
    path('products/', views.UserProductListView.as_view(), name='list-products'),
    path('products/<int:pk>/', views.UserProductDetailView.as_view(), name='product-detail'),
    
    # User dashboard
    path('dashboard/', views.user_dashboard, name='user-dashboard'),
    
    # Admin endpoints
    path('admin/submissions/', views.AdminSubmissionListView.as_view(), name='admin-submissions'),
    path('admin/submissions/<int:pk>/', views.AdminSubmissionDetailView.as_view(), name='admin-submission-detail'),
    path('admin/submissions/<int:batch_id>/approve-reject/', views.admin_approve_reject_batch, name='admin-approve-reject'),
    path('admin/submissions/<int:batch_id>/bulk-list/', views.bulk_list_products, name='bulk-list-products'),
    path('admin/bulk-operations/<int:pk>/', views.BulkOperationStatusView.as_view(), name='bulk-operation-status'),
    path('admin/products/<int:product_id>/update-status/', views.update_product_status, name='update-product-status'),
    path('admin/dashboard/', views.admin_dashboard, name='admin-dashboard'),
]