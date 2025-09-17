from rest_framework import serializers
from .models import Product, ProductImage, SubmissionBatch, BulkListingOperation
from django.utils import timezone


class ProductImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductImage
        fields = ['id', 'image', 'is_primary', 'order']
        read_only_fields = ['id']


class ProductSerializer(serializers.ModelSerializer):
    images = ProductImageSerializer(many=True, read_only=True)
    uploaded_images = serializers.ListField(
        child=serializers.ImageField(max_length=100000, allow_empty_file=False),
        write_only=True,
        required=False
    )

    class Meta:
        model = Product
        fields = [
            'id', 'title', 'description', 'condition', 'defects',
            'estimated_value', 'min_price_range', 'max_price_range', 'confidence',
            'final_listing_price', 'listing_status', 'sold_platform', 'sold_price', 'sold_at',
            'ebay_listing_id', 'amazon_listing_id', 'ebay_category', 'amazon_category',
            'images', 'uploaded_images', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'listing_status', 'sold_platform', 'sold_price', 'sold_at',
            'ebay_listing_id', 'amazon_listing_id', 'created_at', 'updated_at'
        ]

    def create(self, validated_data):
        uploaded_images = validated_data.pop('uploaded_images', [])
        user = self.context['request'].user
        submission_batch = self.context.get('submission_batch')
        
        product = Product.objects.create(
            user=user,
            submission_batch=submission_batch,
            **validated_data
        )
        
        # Create product images
        for index, image in enumerate(uploaded_images):
            ProductImage.objects.create(
                product=product,
                image=image,
                order=index,
                is_primary=(index == 0)  # First image is primary
            )
        
        return product


class SubmissionBatchSerializer(serializers.ModelSerializer):
    products = ProductSerializer(many=True)
    total_items = serializers.ReadOnlyField()
    total_estimated_value = serializers.ReadOnlyField()

    class Meta:
        model = SubmissionBatch
        fields = [
            'id', 'batch_status', 'full_name', 'email', 'phone', 'pickup_date',
            'pickup_address', 'privacy_policy_accepted', 'admin_notes',
            'approved_by', 'approved_at', 'products', 'total_items', 
            'total_estimated_value', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'batch_status', 'admin_notes', 'approved_by', 'approved_at',
            'created_at', 'updated_at'
        ]

    def create(self, validated_data):
        products_data = validated_data.pop('products')
        user = self.context['request'].user
        
        # Create submission batch
        batch = SubmissionBatch.objects.create(user=user, **validated_data)
        
        # Create products for this batch
        for product_data in products_data:
            product_serializer = ProductSerializer(
                data=product_data, 
                context={'request': self.context['request'], 'submission_batch': batch}
            )
            if product_serializer.is_valid():
                product_serializer.save()
        
        return batch

    def validate_pickup_date(self, value):
        if value <= timezone.now():
            raise serializers.ValidationError("Pickup date must be in the future.")
        return value

    def validate_products(self, value):
        if not value:
            raise serializers.ValidationError("At least one product must be submitted.")
        if len(value) > 50:  # Reasonable limit
            raise serializers.ValidationError("Maximum 50 products per submission.")
        return value


class SubmissionBatchListSerializer(serializers.ModelSerializer):
    """Simplified serializer for listing submissions"""
    total_items = serializers.ReadOnlyField()
    total_estimated_value = serializers.ReadOnlyField()

    class Meta:
        model = SubmissionBatch
        fields = [
            'id', 'batch_status', 'full_name', 'email', 'pickup_date',
            'total_items', 'total_estimated_value', 'created_at', 'updated_at'
        ]


class AdminApprovalSerializer(serializers.Serializer):
    """Serializer for admin approval/rejection actions"""
    action = serializers.ChoiceField(choices=['approve', 'reject'])
    notes = serializers.CharField(required=False, allow_blank=True)

    def validate(self, data):
        if data['action'] == 'reject' and not data.get('notes'):
            raise serializers.ValidationError("Notes are required when rejecting a submission.")
        return data


class BulkListingOperationSerializer(serializers.ModelSerializer):
    progress_percentage = serializers.ReadOnlyField()

    class Meta:
        model = BulkListingOperation
        fields = [
            'id', 'operation_type', 'status', 'total_items', 'processed_items',
            'successful_items', 'failed_items', 'progress_percentage', 
            'error_log', 'started_at', 'completed_at'
        ]
        read_only_fields = [
            'id', 'total_items', 'processed_items', 'successful_items', 
            'failed_items', 'started_at', 'completed_at'
        ]


class ProductStatusUpdateSerializer(serializers.Serializer):
    """Serializer for updating product status"""
    platform = serializers.ChoiceField(choices=['EBAY', 'AMAZON'])
    action = serializers.ChoiceField(choices=['sold', 'listed', 'removed'])
    sale_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    listing_id = serializers.CharField(max_length=100, required=False)

    def validate(self, data):
        if data['action'] == 'sold' and not data.get('sale_price'):
            raise serializers.ValidationError("Sale price is required when marking as sold.")
        if data['action'] == 'listed' and not data.get('listing_id'):
            raise serializers.ValidationError("Listing ID is required when marking as listed.")
        return data