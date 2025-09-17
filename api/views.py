from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db import transaction

from .models import Product, SubmissionBatch, BulkListingOperation
from .serializers import (
    SubmissionBatchSerializer, SubmissionBatchListSerializer,
    ProductSerializer, AdminApprovalSerializer,
    BulkListingOperationSerializer, ProductStatusUpdateSerializer
)


class SubmissionBatchCreateView(generics.CreateAPIView):
    """
    Create a new submission batch with multiple products
    """
    serializer_class = SubmissionBatchSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class SubmissionBatchListView(generics.ListAPIView):
    """
    List all submission batches for the authenticated user
    """
    serializer_class = SubmissionBatchListSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return SubmissionBatch.objects.filter(user=self.request.user).order_by('-created_at')


class SubmissionBatchDetailView(generics.RetrieveAPIView):
    """
    Get detailed view of a submission batch including all products
    """
    serializer_class = SubmissionBatchSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return SubmissionBatch.objects.filter(user=self.request.user)


# Admin Views
class AdminSubmissionListView(generics.ListAPIView):
    """
    Admin view to list all pending submissions
    """
    serializer_class = SubmissionBatchListSerializer
    permission_classes = [IsAdminUser]

    def get_queryset(self):
        status_filter = self.request.query_params.get('status', 'PENDING_REVIEW')
        return SubmissionBatch.objects.filter(batch_status=status_filter).order_by('-created_at')


class AdminSubmissionDetailView(generics.RetrieveAPIView):
    """
    Admin detailed view of a submission batch
    """
    serializer_class = SubmissionBatchSerializer
    permission_classes = [IsAdminUser]
    queryset = SubmissionBatch.objects.all()


@api_view(['POST'])
@permission_classes([IsAdminUser])
def admin_approve_reject_batch(request, batch_id):
    """
    Admin endpoint to approve or reject a submission batch
    """
    batch = get_object_or_404(SubmissionBatch, id=batch_id)
    serializer = AdminApprovalSerializer(data=request.data)
    
    if serializer.is_valid():
        action = serializer.validated_data['action']
        notes = serializer.validated_data.get('notes', '')
        
        with transaction.atomic():
            if action == 'approve':
                batch.approve_batch(request.user)
                return Response({
                    'status': 'success',
                    'message': f'Batch {batch_id} approved successfully',
                    'batch_status': batch.batch_status
                })
            elif action == 'reject':
                batch.reject_batch(request.user, notes)
                return Response({
                    'status': 'success',
                    'message': f'Batch {batch_id} rejected',
                    'batch_status': batch.batch_status
                })
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAdminUser])
def bulk_list_products(request, batch_id):
    """
    Admin endpoint to start bulk listing of approved products
    """
    batch = get_object_or_404(SubmissionBatch, id=batch_id)
    
    if batch.batch_status != 'APPROVED':
        return Response({
            'error': 'Batch must be approved before bulk listing'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Create bulk listing operation
    bulk_operation = BulkListingOperation.objects.create(
        submission_batch=batch,
        total_items=batch.total_items
    )
    
    # Update batch status
    batch.batch_status = 'PROCESSING'
    batch.save()
    
    # Here you would trigger the actual listing process
    # For now, we'll just mark it as started
    bulk_operation.status = 'IN_PROGRESS'
    bulk_operation.save()
    
    return Response({
        'status': 'success',
        'message': f'Bulk listing started for batch {batch_id}',
        'operation_id': bulk_operation.id
    })


class BulkOperationStatusView(generics.RetrieveAPIView):
    """
    Check the status of a bulk listing operation
    """
    serializer_class = BulkListingOperationSerializer
    permission_classes = [IsAdminUser]
    queryset = BulkListingOperation.objects.all()


@api_view(['POST'])
@permission_classes([IsAdminUser])
def update_product_status(request, product_id):
    """
    Update product status (sold, listed, removed)
    """
    product = get_object_or_404(Product, id=product_id)
    serializer = ProductStatusUpdateSerializer(data=request.data)
    
    if serializer.is_valid():
        platform = serializer.validated_data['platform']
        action = serializer.validated_data['action']
        
        if action == 'sold':
            sale_price = serializer.validated_data.get('sale_price')
            product.mark_sold(platform, sale_price)
            message = f'Product marked as sold on {platform}'
            
        elif action == 'listed':
            listing_id = serializer.validated_data.get('listing_id')
            if platform == 'EBAY':
                product.ebay_listing_id = listing_id
            elif platform == 'AMAZON':
                product.amazon_listing_id = listing_id
            product.listing_status = 'LISTED'
            product.save()
            message = f'Product marked as listed on {platform}'
            
        elif action == 'removed':
            product.listing_status = 'REMOVED'
            product.save()
            message = f'Product removed from {platform}'
        
        return Response({
            'status': 'success',
            'message': message,
            'product_status': product.listing_status
        })
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# User-facing views
class UserProductListView(generics.ListAPIView):
    """
    List all products for the authenticated user
    """
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Product.objects.filter(user=self.request.user).order_by('-created_at')


class UserProductDetailView(generics.RetrieveAPIView):
    """
    Get detailed view of a specific product
    """
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Product.objects.filter(user=self.request.user)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_dashboard(request):
    """
    Dashboard endpoint providing summary data for the user
    """
    user = request.user
    
    # Get user's submission batches summary
    batches = SubmissionBatch.objects.filter(user=user)
    products = Product.objects.filter(user=user)
    
    dashboard_data = {
        'total_submissions': batches.count(),
        'pending_submissions': batches.filter(batch_status='PENDING_REVIEW').count(),
        'approved_submissions': batches.filter(batch_status='APPROVED').count(),
        'total_products': products.count(),
        'listed_products': products.filter(listing_status='LISTED').count(),
        'sold_products': products.filter(
            listing_status__in=['EBAY_SOLD', 'AMAZON_SOLD']
        ).count(),
        'total_earnings': sum(
            p.sold_price for p in products.filter(sold_price__isnull=False)
        ) or 0,
        'recent_submissions': SubmissionBatchListSerializer(
            batches.order_by('-created_at')[:5], many=True
        ).data
    }
    
    return Response(dashboard_data)


@api_view(['GET'])
@permission_classes([IsAdminUser])
def admin_dashboard(request):
    """
    Admin dashboard with system-wide statistics
    """
    batches = SubmissionBatch.objects.all()
    products = Product.objects.all()
    
    dashboard_data = {
        'pending_reviews': batches.filter(batch_status='PENDING_REVIEW').count(),
        'approved_batches': batches.filter(batch_status='APPROVED').count(),
        'total_products': products.count(),
        'listed_products': products.filter(listing_status='LISTED').count(),
        'sold_products': products.filter(
            listing_status__in=['EBAY_SOLD', 'AMAZON_SOLD']
        ).count(),
        'total_revenue': sum(
            p.sold_price for p in products.filter(sold_price__isnull=False)
        ) or 0,
        'recent_submissions': SubmissionBatchListSerializer(
            batches.order_by('-created_at')[:10], many=True
        ).data,
        'active_bulk_operations': BulkListingOperation.objects.filter(
            status='IN_PROGRESS'
        ).count()
    }
    
    return Response(dashboard_data)
