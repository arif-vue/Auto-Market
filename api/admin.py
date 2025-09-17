from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import Product, ProductImage, SubmissionBatch, BulkListingOperation


class ProductImageInline(admin.TabularInline):
    model = ProductImage
    extra = 0
    readonly_fields = ('image_preview',)

    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 50px;" />',
                obj.image.url
            )
        return "No image"
    image_preview.short_description = "Preview"


class ProductInline(admin.TabularInline):
    model = Product
    extra = 0
    readonly_fields = ('title', 'condition', 'estimated_value', 'listing_status')
    fields = ('title', 'condition', 'estimated_value', 'listing_status', 'final_listing_price')
    can_delete = False


@admin.register(SubmissionBatch)
class SubmissionBatchAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'full_name', 'email', 'batch_status', 'total_items', 
        'total_estimated_value', 'created_at'
    ]
    list_filter = ['batch_status', 'created_at', 'approved_at']
    search_fields = ['full_name', 'email', 'phone']
    readonly_fields = ['user', 'created_at', 'updated_at', 'total_items', 'total_estimated_value']
    inlines = [ProductInline]
    
    fieldsets = (
        ('Contact Information', {
            'fields': ('user', 'full_name', 'email', 'phone')
        }),
        ('Pickup Details', {
            'fields': ('pickup_date', 'pickup_address', 'privacy_policy_accepted')
        }),
        ('Status & Approval', {
            'fields': ('batch_status', 'admin_notes', 'approved_by', 'approved_at')
        }),
        ('Summary', {
            'fields': ('total_items', 'total_estimated_value')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def get_readonly_fields(self, request, obj=None):
        readonly = list(self.readonly_fields)
        if obj and obj.batch_status in ['APPROVED', 'REJECTED']:
            readonly.extend(['batch_status'])
        return readonly

    actions = ['approve_batches', 'reject_batches']

    def approve_batches(self, request, queryset):
        count = 0
        for batch in queryset.filter(batch_status='PENDING_REVIEW'):
            batch.approve_batch(request.user)
            count += 1
        self.message_user(request, f"Approved {count} batches.")
    approve_batches.short_description = "Approve selected batches"

    def reject_batches(self, request, queryset):
        count = 0
        for batch in queryset.filter(batch_status='PENDING_REVIEW'):
            batch.reject_batch(request.user, "Bulk rejection from admin")
            count += 1
        self.message_user(request, f"Rejected {count} batches.")
    reject_batches.short_description = "Reject selected batches"


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = [
        'title', 'user', 'submission_batch_link', 'condition', 
        'estimated_value', 'final_listing_price', 'listing_status', 'created_at'
    ]
    list_filter = [
        'listing_status', 'condition', 'confidence', 'created_at',
        'submission_batch__batch_status'
    ]
    search_fields = ['title', 'description', 'user__email']
    readonly_fields = [
        'user', 'submission_batch', 'estimated_value', 'min_price_range', 
        'max_price_range', 'confidence', 'created_at', 'updated_at'
    ]
    inlines = [ProductImageInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'submission_batch', 'title', 'description', 'condition', 'defects')
        }),
        ('AI Pricing Analysis', {
            'fields': ('estimated_value', 'min_price_range', 'max_price_range', 'confidence'),
            'classes': ('collapse',)
        }),
        ('Final Pricing', {
            'fields': ('final_listing_price',)
        }),
        ('Platform Information', {
            'fields': (
                'ebay_listing_id', 'amazon_listing_id', 
                'ebay_category', 'amazon_category'
            ),
            'classes': ('collapse',)
        }),
        ('Status & Sales', {
            'fields': ('listing_status', 'sold_platform', 'sold_price', 'sold_at')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def submission_batch_link(self, obj):
        if obj.submission_batch:
            url = reverse('admin:api_submissionbatch_change', args=[obj.submission_batch.id])
            return format_html('<a href="{}">{}</a>', url, obj.submission_batch.id)
        return "N/A"
    submission_batch_link.short_description = "Submission Batch"

    actions = ['mark_as_listed', 'mark_as_removed']

    def mark_as_listed(self, request, queryset):
        count = queryset.filter(listing_status='APPROVED').update(listing_status='LISTED')
        self.message_user(request, f"Marked {count} products as listed.")
    mark_as_listed.short_description = "Mark as listed on platforms"

    def mark_as_removed(self, request, queryset):
        count = queryset.update(listing_status='REMOVED')
        self.message_user(request, f"Marked {count} products as removed.")
    mark_as_removed.short_description = "Mark as removed from platforms"


@admin.register(ProductImage)
class ProductImageAdmin(admin.ModelAdmin):
    list_display = ['product', 'is_primary', 'order', 'image_preview']
    list_filter = ['is_primary', 'product__listing_status']
    search_fields = ['product__title']
    readonly_fields = ['image_preview']

    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 100px; max-width: 100px;" />',
                obj.image.url
            )
        return "No image"
    image_preview.short_description = "Preview"


@admin.register(BulkListingOperation)
class BulkListingOperationAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'submission_batch', 'operation_type', 'status', 
        'progress_display', 'started_at', 'completed_at'
    ]
    list_filter = ['status', 'operation_type', 'started_at']
    readonly_fields = [
        'submission_batch', 'total_items', 'processed_items', 
        'successful_items', 'failed_items', 'progress_percentage',
        'started_at', 'completed_at'
    ]
    
    fieldsets = (
        ('Operation Details', {
            'fields': ('submission_batch', 'operation_type', 'status')
        }),
        ('Progress Tracking', {
            'fields': (
                'total_items', 'processed_items', 'successful_items', 
                'failed_items', 'progress_percentage'
            )
        }),
        ('Error Information', {
            'fields': ('error_log',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('started_at', 'completed_at')
        }),
    )

    def progress_display(self, obj):
        percentage = obj.progress_percentage
        if percentage == 0:
            return "Not started"
        return f"{percentage:.1f}% ({obj.processed_items}/{obj.total_items})"
    progress_display.short_description = "Progress"

    def has_add_permission(self, request):
        return False  # Operations are created programmatically


# Customize admin site
admin.site.site_header = "Auto Market Administration"
admin.site.site_title = "Auto Market Admin"
admin.site.index_title = "Welcome to Auto Market Administration"
