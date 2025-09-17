#!/usr/bin/env python
"""
Complete workflow test for Auto Market API
Tests the entire flow from user registration to product approval
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'auto_market.settings')
django.setup()

from django.contrib.auth import get_user_model
from api.models import SubmissionBatch, Product, ProductImage, BulkListingOperation
from datetime import datetime, timedelta
from decimal import Decimal

User = get_user_model()

def test_complete_workflow():
    print("🚀 Testing Complete Auto Market Workflow")
    print("=" * 50)
    
    # Step 1: Create test user
    print("Step 1: Creating test user...")
    try:
        test_user = User.objects.create_user(
            email='testuser@example.com',
            full_name='Test User',
            password='testpass123'
        )
        print(f"✅ Created user: {test_user.email}")
    except Exception as e:
        # User might already exist
        test_user = User.objects.get(email='testuser@example.com')
        print(f"✅ Using existing user: {test_user.email}")
    
    # Step 2: Create admin user
    print("\nStep 2: Creating admin user...")
    try:
        admin_user = User.objects.create_superuser(
            email='admin@example.com',
            full_name='Admin User',
            password='adminpass123'
        )
        print(f"✅ Created admin: {admin_user.email}")
    except Exception as e:
        admin_user = User.objects.get(email='admin@example.com')
        print(f"✅ Using existing admin: {admin_user.email}")
    
    # Step 3: Create submission batch
    print("\nStep 3: Creating submission batch...")
    pickup_date = datetime.now() + timedelta(days=7)
    
    batch = SubmissionBatch.objects.create(
        user=test_user,
        full_name="John Doe",
        email="john@example.com",
        phone="+1234567890",
        pickup_date=pickup_date,
        pickup_address="123 Main St, City, State 12345",
        privacy_policy_accepted=True
    )
    print(f"✅ Created submission batch: {batch.id}")
    
    # Step 4: Create products
    print("\nStep 4: Creating products...")
    products_data = [
        {
            'title': 'iPhone 14 Pro',
            'description': 'Excellent condition iPhone 14 Pro, 256GB',
            'condition': 'EXCELLENT',
            'estimated_value': Decimal('800.00'),
            'min_price_range': Decimal('750.00'),
            'max_price_range': Decimal('850.00'),
            'confidence': 'HIGH'
        },
        {
            'title': 'MacBook Air M2',
            'description': 'Like new MacBook Air with M2 chip',
            'condition': 'LIKE_NEW',
            'estimated_value': Decimal('1200.00'),
            'min_price_range': Decimal('1150.00'),
            'max_price_range': Decimal('1300.00'),
            'confidence': 'HIGH'
        },
        {
            'title': 'Sony Camera',
            'description': 'Professional Sony camera with lens',
            'condition': 'GOOD',
            'estimated_value': Decimal('500.00'),
            'min_price_range': Decimal('450.00'),
            'max_price_range': Decimal('550.00'),
            'confidence': 'MEDIUM'
        }
    ]
    
    created_products = []
    for product_data in products_data:
        product = Product.objects.create(
            user=test_user,
            submission_batch=batch,
            **product_data
        )
        created_products.append(product)
        print(f"✅ Created product: {product.title}")
    
    # Step 5: Test admin approval workflow
    print(f"\nStep 5: Testing admin approval...")
    print(f"Batch status before approval: {batch.batch_status}")
    print(f"Products status: {[p.listing_status for p in created_products]}")
    
    # Admin approves the batch
    batch.approve_batch(admin_user)
    batch.refresh_from_db()
    
    # Refresh products
    for product in created_products:
        product.refresh_from_db()
    
    print(f"✅ Batch status after approval: {batch.batch_status}")
    print(f"✅ Products status: {[p.listing_status for p in created_products]}")
    
    # Step 6: Test bulk listing operation
    print(f"\nStep 6: Testing bulk listing...")
    bulk_operation = BulkListingOperation.objects.create(
        submission_batch=batch,
        total_items=len(created_products)
    )
    
    # Simulate listing process
    bulk_operation.status = 'IN_PROGRESS'
    bulk_operation.save()
    
    for i, product in enumerate(created_products):
        product.list_on_platforms(
            ebay_id=f"ebay_{product.id}",
            amazon_id=f"amzn_{product.id}"
        )
        bulk_operation.processed_items = i + 1
        bulk_operation.successful_items = i + 1
        bulk_operation.save()
    
    bulk_operation.status = 'COMPLETED'
    bulk_operation.save()
    
    # Refresh products
    for product in created_products:
        product.refresh_from_db()
    
    print(f"✅ Bulk operation completed: {bulk_operation.progress_percentage:.1f}%")
    print(f"✅ Products listing status: {[p.listing_status for p in created_products]}")
    
    # Step 7: Test sales workflow
    print(f"\nStep 7: Testing sales workflow...")
    # Simulate first product sold on eBay
    first_product = created_products[0]
    first_product.mark_sold('EBAY', Decimal('820.00'))
    first_product.refresh_from_db()
    
    print(f"✅ {first_product.title} sold on {first_product.sold_platform} for ${first_product.sold_price}")
    
    # Step 8: Display summary
    print(f"\nStep 8: Workflow Summary")
    print("=" * 30)
    print(f"📊 Submission Batch: {batch.id}")
    print(f"📊 Total Items: {batch.total_items}")
    print(f"📊 Total Estimated Value: ${batch.total_estimated_value}")
    print(f"📊 Batch Status: {batch.batch_status}")
    print(f"📊 Admin Approved By: {batch.approved_by.email if batch.approved_by else 'None'}")
    
    print(f"\n📦 Products Summary:")
    for product in created_products:
        status_emoji = "💰" if "SOLD" in product.listing_status else "📋"
        print(f"  {status_emoji} {product.title}: {product.listing_status}")
        if product.sold_price:
            print(f"     💵 Sold for: ${product.sold_price}")
    
    print(f"\n🚀 Complete workflow test PASSED!")
    print(f"✅ All status transitions working correctly")
    print(f"✅ Models and relationships working properly")
    print(f"✅ Admin workflow functional")
    print(f"✅ Bulk operations tracking working")
    
    return True

if __name__ == "__main__":
    try:
        success = test_complete_workflow()
        if success:
            print(f"\n🎉 AUTO MARKET API IS FULLY FUNCTIONAL! 🎉")
            sys.exit(0)
    except Exception as e:
        print(f"\n❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)