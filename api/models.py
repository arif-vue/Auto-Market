from django.db import models
from django.conf import settings

class Product(models.Model):
    LISTING_STATUS_CHOICES = (
        ('PENDING', 'Pending Approval'),
        ('APPROVED', 'Approved for Listing'),
        ('LISTED', 'Listed on Both Platforms'),
        ('EBAY_SOLD', 'Sold on eBay'),
        ('AMAZON_SOLD', 'Sold on Amazon'),
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
    
    # Basic Information
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()
    condition = models.CharField(max_length=20, choices=ITEM_CONDITION_CHOICES, default="GOOD")
    defects = models.TextField(blank=True)
    
    # Pricing will detect by AI
    estimated_value = models.DecimalField(max_digits=10, decimal_places=2)
    min_price_range = models.DecimalField(max_digits=10, decimal_places=2)
    max_price_range = models.DecimalField(max_digits=10, decimal_places=2)
    confidence = models.CharField(max_length=10, choices=CONFIDENCE_CHOICES, default="MEDIUM")
    
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
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    def mark_sold(self, platform):
        """Mark item as sold on specific platform and update status accordingly"""
        if platform == 'EBAY':
            self.listing_status = 'EBAY_SOLD'
            # Logic to delete Amazon listing would go here
        elif platform == 'AMAZON':
            self.listing_status = 'AMAZON_SOLD'
            # Logic to delete eBay listing would go here
        self.save()


class ProductImage(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='product_images/')
    is_primary = models.BooleanField(default=False)
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ['order']


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