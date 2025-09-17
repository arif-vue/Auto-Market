from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils import timezone


class SubmissionBatch(models.Model):
    """Groups multiple products submitted together with contact info"""
    BATCH_STATUS_CHOICES = (
        ('PENDING_REVIEW', 'Pending Admin Review'),
        ('APPROVED', 'Approved for Processing'),
        ('REJECTED', 'Rejected'),
        ('PROCESSING', 'Items Being Listed'),
        ('COMPLETED', 'All Items Processed'),
    )
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    batch_status = models.CharField(max_length=20, choices=BATCH_STATUS_CHOICES, default='PENDING_REVIEW')
    
    # Contact Information
    full_name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    pickup_date = models.DateTimeField()
    pickup_address = models.TextField()
    privacy_policy_accepted = models.BooleanField(default=False)
    
    # Admin notes
    admin_notes = models.TextField(blank=True, help_text="Admin notes for approval/rejection")
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='approved_batches'
    )
    approved_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Batch {self.id} - {self.full_name} ({self.batch_status})"

    @property
    def total_items(self):
        return self.products.count()

    @property
    def total_estimated_value(self):
        return sum(product.estimated_value for product in self.products.all())

    def approve_batch(self, admin_user):
        """Admin approves the entire batch"""
        self.batch_status = 'APPROVED'
        self.approved_by = admin_user
        self.approved_at = timezone.now()
        self.save()
        
        # Update all products in batch
        self.products.update(listing_status='APPROVED')

    def reject_batch(self, admin_user, reason=""):
        """Admin rejects the entire batch"""
        self.batch_status = 'REJECTED'
        self.approved_by = admin_user
        self.admin_notes = reason
        self.save()


class Product(models.Model):
    LISTING_STATUS_CHOICES = (
        ('PENDING', 'Pending Approval'),
        ('APPROVED', 'Approved for Listing'),
        ('LISTED', 'Listed on Both Platforms'),
        ('EBAY_SOLD', 'Sold on eBay'),
        ('AMAZON_SOLD', 'Sold on Amazon'),
        ('REMOVED', 'Removed from Listings'),
    )
    
    ITEM_CONDITION_CHOICES = (
        ("NEW", "New"),
        ("LIKE_NEW", "Like New"),
        ("EXCELLENT", "Excellent"),
        ("GOOD", "Good"),
        ("FAIR", "Fair"),
        ("POOR", "Poor")
    )
    
    CONFIDENCE_CHOICES = (
        ("HIGH", "High"),
        ("MEDIUM", "Medium"),
        ("LOW", "Low")
    )
    
    # Link to submission batch
    submission_batch = models.ForeignKey(
        SubmissionBatch, 
        on_delete=models.CASCADE, 
        related_name='products',
        null=True, 
        blank=True
    )
    
    # Basic Information
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()
    condition = models.CharField(max_length=20, choices=ITEM_CONDITION_CHOICES, default="GOOD")
    defects = models.TextField(blank=True)
    
    # AI-detected pricing
    estimated_value = models.DecimalField(max_digits=10, decimal_places=2)
    min_price_range = models.DecimalField(max_digits=10, decimal_places=2)
    max_price_range = models.DecimalField(max_digits=10, decimal_places=2)
    confidence = models.CharField(max_length=10, choices=CONFIDENCE_CHOICES, default="MEDIUM")
    
    # Final listing price (after admin review)
    final_listing_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Platform-specific fields
    ebay_listing_id = models.CharField(max_length=100, blank=True, null=True)
    amazon_listing_id = models.CharField(max_length=100, blank=True, null=True)
    ebay_category = models.CharField(max_length=50, blank=True, null=True)
    amazon_category = models.CharField(max_length=50, blank=True, null=True)
    
    # Status tracking
    listing_status = models.CharField(
        max_length=20, 
        choices=LISTING_STATUS_CHOICES, 
        default='PENDING'
    )
    
    # Sale information
    sold_platform = models.CharField(max_length=20, blank=True, null=True)
    sold_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    sold_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} - {self.get_listing_status_display()}"

    def mark_sold(self, platform, sale_price=None):
        """Mark item as sold on specific platform and update status accordingly"""
        from django.utils import timezone
        
        if platform.upper() == 'EBAY':
            self.listing_status = 'EBAY_SOLD'
            self.sold_platform = 'EBAY'
        elif platform.upper() == 'AMAZON':
            self.listing_status = 'AMAZON_SOLD'
            self.sold_platform = 'AMAZON'
        
        if sale_price:
            self.sold_price = sale_price
        self.sold_at = timezone.now()
        self.save()

    def list_on_platforms(self, ebay_id=None, amazon_id=None):
        """Mark as listed on both platforms"""
        self.listing_status = 'LISTED'
        if ebay_id:
            self.ebay_listing_id = ebay_id
        if amazon_id:
            self.amazon_listing_id = amazon_id
        self.save()


class ProductImage(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='product_images/')
    is_primary = models.BooleanField(default=False)
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"Image {self.order} for {self.product.title}"


# Legacy model - keeping for backward compatibility but functionality moved to SubmissionBatch
class SellerContactInfo(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    pickup_date = models.DateTimeField()
    pickup_address = models.TextField()
    products = models.ManyToManyField(Product, related_name="contact_info")
    privacy_policy_accepted = models.BooleanField(default=False)
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.full_name} - {self.email}"

    class Meta:
        verbose_name = "Legacy Seller Contact Info"
        verbose_name_plural = "Legacy Seller Contact Info"


class BulkListingOperation(models.Model):
    """Track bulk listing operations"""
    OPERATION_STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('IN_PROGRESS', 'In Progress'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
        ('PARTIAL', 'Partially Completed'),
    )
    
    submission_batch = models.ForeignKey(SubmissionBatch, on_delete=models.CASCADE)
    operation_type = models.CharField(max_length=20, default='BULK_LIST')
    status = models.CharField(max_length=20, choices=OPERATION_STATUS_CHOICES, default='PENDING')
    
    # Progress tracking
    total_items = models.IntegerField(default=0)
    processed_items = models.IntegerField(default=0)
    successful_items = models.IntegerField(default=0)
    failed_items = models.IntegerField(default=0)
    
    # Error tracking
    error_log = models.TextField(blank=True)
    
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Bulk operation for Batch {self.submission_batch.id} - {self.status}"

    @property
    def progress_percentage(self):
        if self.total_items == 0:
            return 0
        return (self.processed_items / self.total_items) * 100