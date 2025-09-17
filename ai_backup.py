"""
Django Integration Guide:
1. Add this module to your Django project
2. Use the services classes in your Django views
3. Configure settings in Django settings.py
4. Use the models for database integration
5. Implement the API endpoints in your Django views/viewsets
"""

import os
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import quote_plus
import tempfile
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import logging
from datetime import datetime
import warnings
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*image processor.*", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== DJANGO-COMPATIBLE CONFIGURATION =====

class AutoMarketConfig:
    """
    Django-compatible configuration class
    Can be integrated with Django settings
    """
    
    # Default configuration - override in Django settings.py
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    TEMP_IMAGES_FOLDER = 'temp_images'
    
    # AI Model Configuration
    DEFAULT_MODEL_NAME = "google/efficientnet-b0"
    FALLBACK_MODEL_NAME = "microsoft/resnet-50"
    DAMAGE_DETECTOR_MODEL = "facebook/detr-resnet-50"
    
    # Price Scraping Configuration
    MAX_PRICE_RESULTS = 25
    REQUEST_TIMEOUT = 10
    DELAY_RANGE = (1, 3)
    
    # Regional Pricing Configuration
    REGIONAL_MULTIPLIERS = {
        # US Major Cities
        'new_york': 1.25, 'san_francisco': 1.30, 'los_angeles': 1.15,
        'chicago': 1.10, 'boston': 1.20, 'seattle': 1.15, 'miami': 1.05,
        'atlanta': 1.00, 'dallas': 0.95, 'houston': 0.95, 'phoenix': 0.90,
        'philadelphia': 1.05, 'washington_dc': 1.15,
        
        # US States (average)
        'california': 1.15, 'new_york_state': 1.15, 'massachusetts': 1.15,
        'washington_state': 1.10, 'connecticut': 1.10, 'new_jersey': 1.10,
        'maryland': 1.05, 'virginia': 1.00, 'texas': 0.95, 'florida': 1.00,
        'arizona': 0.90, 'nevada': 0.95, 'oregon': 1.05,
        
        # International (examples)
        'london': 1.20, 'tokyo': 1.15, 'sydney': 1.10, 'toronto': 1.05,
        'vancouver': 1.10, 'paris': 1.15, 'berlin': 1.00, 'amsterdam': 1.10
    }
    
    # Market Data Configuration
    MIN_DATA_POINTS_HIGH_CONFIDENCE = 15
    MIN_DATA_POINTS_MEDIUM_CONFIDENCE = 8
    MIN_DATA_POINTS_LOW_CONFIDENCE = 3
    
    @classmethod
    def from_django_settings(cls, settings):
        """
        Create configuration from Django settings
        Usage in Django:
            from django.conf import settings
            config = AutoMarketConfig.from_django_settings(settings)
        """
        config = cls()
        
        # Override with Django settings if available
        if hasattr(settings, 'AUTO_MARKET_ALLOWED_EXTENSIONS'):
            config.ALLOWED_EXTENSIONS = settings.AUTO_MARKET_ALLOWED_EXTENSIONS
        
        if hasattr(settings, 'AUTO_MARKET_TEMP_FOLDER'):
            config.TEMP_IMAGES_FOLDER = settings.AUTO_MARKET_TEMP_FOLDER
            
        if hasattr(settings, 'AUTO_MARKET_MODEL_NAME'):
            config.DEFAULT_MODEL_NAME = settings.AUTO_MARKET_MODEL_NAME
            
        if hasattr(settings, 'AUTO_MARKET_REGIONAL_MULTIPLIERS'):
            config.REGIONAL_MULTIPLIERS.update(settings.AUTO_MARKET_REGIONAL_MULTIPLIERS)
        
        return config

# ===== DJANGO-COMPATIBLE DATA MODELS =====

@dataclass
class ItemCondition:
    """Data class for item condition analysis results"""
    condition_score: float
    detected_condition: str
    defects_detected: List[str]
    confidence: float
    analysis_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Django JSON responses"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ItemCondition':
        """Create from dictionary for Django form processing"""
        return cls(**data)

@dataclass
class PriceEstimate:
    """Data class for price estimation results"""
    estimated_value: int
    range_low: int
    range_high: int
    confidence: str
    ai_detected_condition: str
    condition_confidence: float
    market_data_points: int
    regional_multiplier: float
    timestamp: str
    note: str
    item_name: Optional[str] = None
    condition_reported: Optional[str] = None
    condition_detection_used: bool = False
    images_analyzed: int = 0
    data_points_found: int = 0
    search_success: bool = False
    message: Optional[str] = None
    market_analysis: Optional[Dict[str, Any]] = None
    price_validation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Django JSON responses"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceEstimate':
        """Create from dictionary for Django form processing"""
        return cls(**data)

@dataclass
class SearchResult:
    """Data class for search results"""
    price: float
    source: str
    condition: str
    date_sold: Optional[str]
    title: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ValuationRequest:
    """Data class for valuation requests - Django form compatible"""
    item_name: str
    description: str
    condition: str
    issues: str = ""
    user_location: Optional[str] = None
    image_urls: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_django_request(cls, request) -> 'ValuationRequest':
        """
        Create from Django request object
        Usage in Django views:
            valuation_request = ValuationRequest.from_django_request(request)
        """
        return cls(
            item_name=request.POST.get('itemName', '').strip(),
            description=request.POST.get('description', '').strip(),
            condition=request.POST.get('condition', '').strip(),
            issues=request.POST.get('issues', '').strip(),
            user_location=request.POST.get('userLocation'),
            image_urls=request.POST.getlist('imageUrls') if hasattr(request.POST, 'getlist') else []
        )

# ===== DJANGO-COMPATIBLE PRICING FUNCTIONS =====

def get_estimated_value(item_name: str, description: str, condition: str, 
                       issues: str = "", image_paths: List[str] = None, 
                       user_location: str = None) -> int:
    """
    Get AI-generated estimated value for an item
    
    Args:
        item_name: Name of the item
        description: Item description
        condition: Item condition (Excellent, Good, Fair, Poor)
        issues: Any issues or defects
        image_paths: List of image paths for analysis
        user_location: User location for regional pricing
    
    Returns:
        int: Estimated value in dollars
    
    Usage in Django:
        estimated_value = get_estimated_value("MacBook Pro", "2019 model", "Good")
    """
    try:
        from .services import ItemValuationService
        service = ItemValuationService()
        result = service.estimate_value(
            item_name=item_name,
            description=description,
            condition=condition,
            issues=issues,
            image_paths=image_paths or [],
            user_location=user_location
        )
        return result.estimated_value
    except Exception as e:
        logger.error(f"Error getting estimated value: {e}")
        return 0

def get_min_price_range(item_name: str, description: str, condition: str, 
                       confidence_level: str = "medium") -> int:
    """
    Get minimum price range for an item
    
    Args:
        item_name: Name of the item
        description: Item description
        condition: Item condition
        confidence_level: "high", "medium", or "low"
    
    Returns:
        int: Minimum price in dollars
    
    Usage in Django:
        min_price = get_min_price_range("iPhone 12", "64GB", "Good")
    """
    try:
        from .services import ItemValuationService
        service = ItemValuationService()
        result = service.estimate_value(item_name, description, condition)
        
        # Calculate minimum based on confidence level
        confidence_multipliers = {
            "high": 0.85,    # 85% of estimated value
            "medium": 0.75,  # 75% of estimated value  
            "low": 0.65      # 65% of estimated value
        }
        
        multiplier = confidence_multipliers.get(confidence_level.lower(), 0.75)
        return int(result.estimated_value * multiplier)
        
    except Exception as e:
        logger.error(f"Error getting min price range: {e}")
        return 0

def get_max_price_range(item_name: str, description: str, condition: str, 
                       confidence_level: str = "medium") -> int:
    """
    Get maximum price range for an item
    
    Args:
        item_name: Name of the item
        description: Item description
        condition: Item condition
        confidence_level: "high", "medium", or "low"
    
    Returns:
        int: Maximum price in dollars
    
    Usage in Django:
        max_price = get_max_price_range("iPhone 12", "64GB", "Good")
    """
    try:
        from .services import ItemValuationService
        service = ItemValuationService()
        result = service.estimate_value(item_name, description, condition)
        
        # Calculate maximum based on confidence level
        confidence_multipliers = {
            "high": 1.15,    # 115% of estimated value
            "medium": 1.25,  # 125% of estimated value
            "low": 1.35      # 135% of estimated value
        }
        
        multiplier = confidence_multipliers.get(confidence_level.lower(), 1.25)
        return int(result.estimated_value * multiplier)
        
    except Exception as e:
        logger.error(f"Error getting max price range: {e}")
        return 0

def get_confidence_level(item_name: str, description: str, condition: str, 
                        image_paths: List[str] = None) -> str:
    """
    Get confidence level for pricing estimate
    
    Args:
        item_name: Name of the item
        description: Item description
        condition: Item condition
        image_paths: List of image paths for analysis
    
    Returns:
        str: Confidence level ("High", "Medium", "Low")
    
    Usage in Django:
        confidence = get_confidence_level("MacBook Pro", "2019 model", "Good")
    """
    try:
        from .services import ItemValuationService
        service = ItemValuationService()
        result = service.estimate_value(
            item_name=item_name,
            description=description,
            condition=condition,
            image_paths=image_paths or []
        )
        return result.confidence
        
    except Exception as e:
        logger.error(f"Error getting confidence level: {e}")
        return "Low"

def get_pricing_summary(item_name: str, description: str, condition: str, 
                       issues: str = "", image_paths: List[str] = None, 
                       user_location: str = None) -> Dict[str, Any]:
    """
    Get complete pricing summary with all pricing functions
    
    Args:
        item_name: Name of the item
        description: Item description
        condition: Item condition
        issues: Any issues or defects
        image_paths: List of image paths
        user_location: User location
    
    Returns:
        dict: Complete pricing summary
    
    Usage in Django:
        pricing = get_pricing_summary("iPhone 12", "64GB", "Good")
        # Returns: {
        #     'estimated_value': 400,
        #     'min_price_range': 300,
        #     'max_price_range': 500,
        #     'confidence': 'Medium'
        # }
    """
    try:
        estimated_value = get_estimated_value(
            item_name, description, condition, issues, image_paths, user_location
        )
        
        confidence = get_confidence_level(item_name, description, condition, image_paths)
        
        min_price = get_min_price_range(item_name, description, condition, confidence.lower())
        max_price = get_max_price_range(item_name, description, condition, confidence.lower())
        
        return {
            'estimated_value': estimated_value,
            'min_price_range': min_price,
            'max_price_range': max_price,
            'confidence': confidence,
            'item_name': item_name,
            'condition': condition,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting pricing summary: {e}")
        return {
            'estimated_value': 0,
            'min_price_range': 0,
            'max_price_range': 0,
            'confidence': 'Low',
            'error': str(e)
        }

# ===== DJANGO MODEL COMPATIBLE FUNCTIONS =====

def create_valuation_record(item_name: str, description: str, condition: str,
                           estimated_value: int, min_price: int, max_price: int,
                           confidence: str, user_id: int = None) -> Dict[str, Any]:
    """
    Create a valuation record compatible with Django models
    
    Usage in Django:
        from django.contrib.auth.models import User
        from myapp.models import ValuationRecord
        
        record_data = create_valuation_record(
            item_name="iPhone 12",
            description="64GB",
            condition="Good",
            estimated_value=400,
            min_price=300,
            max_price=500,
            confidence="Medium",
            user_id=request.user.id
        )
        
        valuation = ValuationRecord.objects.create(**record_data)
    """
    return {
        'item_name': item_name,
        'description': description,
        'condition': condition,
        'estimated_value': estimated_value,
        'min_price_range': min_price,
        'max_price_range': max_price,
        'confidence': confidence,
        'user_id': user_id,
        'created_at': datetime.now(),
        'updated_at': datetime.now()
    }

# ===== DJANGO FORM HELPER FUNCTIONS =====

def validate_pricing_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pricing form data for Django forms
    
    Usage in Django:
        def clean(self):
            cleaned_data = super().clean()
            validation_result = validate_pricing_form(cleaned_data)
            if not validation_result['is_valid']:
                raise forms.ValidationError(validation_result['errors'])
            return cleaned_data
    """
    errors = []
    
    # Required fields validation
    required_fields = ['item_name', 'description', 'condition']
    for field in required_fields:
        if not form_data.get(field, '').strip():
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    # Condition validation
    valid_conditions = ['Excellent', 'Good', 'Fair', 'Poor']
    if form_data.get('condition') not in valid_conditions:
        errors.append(f"Condition must be one of: {', '.join(valid_conditions)}")
    
    # Item name length validation
    if len(form_data.get('item_name', '')) > 200:
        errors.append("Item name must be less than 200 characters")
    
    # Description length validation
    if len(form_data.get('description', '')) > 1000:
        errors.append("Description must be less than 1000 characters")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'cleaned_data': form_data if len(errors) == 0 else {}
    }

# ===== DJANGO UTILITY FUNCTIONS =====

class DjangoPricingService:
    """
    Django-specific pricing service class
    Provides a clean interface for Django developers
    
    Usage in Django:
        # In your Django view
        from auto_market_core import DjangoPricingService
        
        pricing_service = DjangoPricingService()
        result = pricing_service.get_item_pricing(
            item_name="iPhone 12",
            description="64GB",
            condition="Good"
        )
    """
    
    def __init__(self, config: AutoMarketConfig = None):
        """Initialize with optional configuration"""
        self.config = config or AutoMarketConfig()
        self._valuation_service = None
    
    @property
    def valuation_service(self):
        """Lazy load valuation service"""
        if self._valuation_service is None:
            try:
                from .services import ItemValuationService
                self._valuation_service = ItemValuationService(self.config)
            except ImportError:
                logger.warning("ItemValuationService not available, using fallback methods")
                self._valuation_service = None
        return self._valuation_service
    
    def get_item_pricing(self, item_name: str, description: str, condition: str, 
                        issues: str = "", image_paths: List[str] = None, 
                        user_location: str = None) -> Dict[str, Any]:
        """
        Get complete item pricing information
        
        Returns:
            dict: Pricing information with estimated_value, min_price_range, 
                  max_price_range, and confidence
        """
        return get_pricing_summary(
            item_name=item_name,
            description=description,
            condition=condition,
            issues=issues,
            image_paths=image_paths,
            user_location=user_location
        )
    
    def estimate_value(self, item_name: str, description: str, condition: str) -> int:
        """Get estimated value only"""
        return get_estimated_value(item_name, description, condition)
    
    def get_price_range(self, item_name: str, description: str, condition: str) -> Tuple[int, int]:
        """Get price range as tuple (min, max)"""
        min_price = get_min_price_range(item_name, description, condition)
        max_price = get_max_price_range(item_name, description, condition)
        return (min_price, max_price)
    
    def get_confidence(self, item_name: str, description: str, condition: str) -> str:
        """Get confidence level"""
        return get_confidence_level(item_name, description, condition)
    
    def validate_form_data(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Django form data"""
        return validate_pricing_form(form_data)
    
    def create_record_data(self, item_name: str, description: str, condition: str,
                          user_id: int = None) -> Dict[str, Any]:
        """Create data dictionary for Django model creation"""
        pricing = self.get_item_pricing(item_name, description, condition)
        
        return create_valuation_record(
            item_name=item_name,
            description=description,
            condition=condition,
            estimated_value=pricing['estimated_value'],
            min_price=pricing['min_price_range'],
            max_price=pricing['max_price_range'],
            confidence=pricing['confidence'],
            user_id=user_id
        )

# ===== DJANGO MODEL SUGGESTIONS =====

def get_django_model_code() -> str:
    """
    Returns Django model code that can be used in models.py
    
    Usage:
        print(get_django_model_code())
        # Copy the output to your Django models.py file
    """
    return '''
# Add this to your Django models.py file

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class ItemValuation(models.Model):
    """Model for storing item valuations"""
    
    CONDITION_CHOICES = [
        ('Excellent', 'Excellent'),
        ('Good', 'Good'),
        ('Fair', 'Fair'),
        ('Poor', 'Poor'),
    ]
    
    CONFIDENCE_CHOICES = [
        ('High', 'High'),
        ('Medium', 'Medium'),
        ('Low', 'Low'),
    ]
    
    # Basic item information
    item_name = models.CharField(max_length=200)
    description = models.TextField(max_length=1000)
    condition = models.CharField(max_length=20, choices=CONDITION_CHOICES)
    issues = models.TextField(blank=True, null=True)
    
    # Pricing information (AI generated)
    estimated_value = models.PositiveIntegerField(validators=[MinValueValidator(0)])
    min_price_range = models.PositiveIntegerField(validators=[MinValueValidator(0)])
    max_price_range = models.PositiveIntegerField(validators=[MinValueValidator(0)])
    confidence = models.CharField(max_length=10, choices=CONFIDENCE_CHOICES)
    
    # Metadata
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Additional fields
    ai_detected_condition = models.CharField(max_length=20, blank=True, null=True)
    market_data_points = models.PositiveIntegerField(default=0)
    regional_multiplier = models.FloatField(default=1.0)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Item Valuation"
        verbose_name_plural = "Item Valuations"
    
    def __str__(self):
        return f"{self.item_name} - ${self.estimated_value}"
    
    @property
    def price_range_text(self):
        return f"${self.min_price_range} - ${self.max_price_range}"

class ValuationImage(models.Model):
    """Model for storing valuation images"""
    valuation = models.ForeignKey(ItemValuation, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='valuations/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    ai_analysis_result = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"Image for {self.valuation.item_name}"
'''

def get_django_form_code() -> str:
    """
    Returns Django form code that can be used in forms.py
    """
    return '''
# Add this to your Django forms.py file

from django import forms
from .models import ItemValuation
from auto_market_core import validate_pricing_form

class ItemValuationForm(forms.ModelForm):
    """Form for item valuation"""
    
    class Meta:
        model = ItemValuation
        fields = ['item_name', 'description', 'condition', 'issues']
        widgets = {
            'item_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter item name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the item in detail'
            }),
            'condition': forms.Select(attrs={
                'class': 'form-control'
            }),
            'issues': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Any issues or defects (optional)'
            }),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        
        # Use the auto_market_core validation
        validation_result = validate_pricing_form({
            'item_name': cleaned_data.get('item_name', ''),
            'description': cleaned_data.get('description', ''),
            'condition': cleaned_data.get('condition', ''),
            'issues': cleaned_data.get('issues', ''),
        })
        
        if not validation_result['is_valid']:
            for error in validation_result['errors']:
                raise forms.ValidationError(error)
        
        return cleaned_data

class ImageUploadForm(forms.Form):
    """Form for uploading images"""
    images = forms.FileField(
        widget=forms.ClearableFileInput(attrs={
            'multiple': True,
            'class': 'form-control'
        }),
        required=False
    )
'''

def get_django_view_code() -> str:
    """
    Returns Django view code examples
    """
    return '''
# Add this to your Django views.py file

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import CreateView, ListView
from django.contrib import messages
import json

from .models import ItemValuation
from .forms import ItemValuationForm, ImageUploadForm
from auto_market_core import DjangoPricingService

class ItemValuationCreateView(CreateView):
    """Create view for item valuations"""
    model = ItemValuation
    form_class = ItemValuationForm
    template_name = 'valuations/create.html'
    success_url = '/valuations/'
    
    def form_valid(self, form):
        # Get pricing information before saving
        pricing_service = DjangoPricingService()
        
        pricing_data = pricing_service.get_item_pricing(
            item_name=form.cleaned_data['item_name'],
            description=form.cleaned_data['description'],
            condition=form.cleaned_data['condition'],
            issues=form.cleaned_data['issues']
        )
        
        # Set pricing fields
        form.instance.estimated_value = pricing_data['estimated_value']
        form.instance.min_price_range = pricing_data['min_price_range']
        form.instance.max_price_range = pricing_data['max_price_range']
        form.instance.confidence = pricing_data['confidence']
        
        if self.request.user.is_authenticated:
            form.instance.user = self.request.user
        
        return super().form_valid(form)

class ItemValuationListView(ListView):
    """List view for item valuations"""
    model = ItemValuation
    template_name = 'valuations/list.html'
    context_object_name = 'valuations'
    paginate_by = 20
    
    def get_queryset(self):
        if self.request.user.is_authenticated:
            return ItemValuation.objects.filter(user=self.request.user)
        return ItemValuation.objects.none()

@csrf_exempt
def ajax_estimate_value(request):
    """AJAX endpoint for quick price estimates"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            pricing_service = DjangoPricingService()
            result = pricing_service.get_item_pricing(
                item_name=data.get('item_name', ''),
                description=data.get('description', ''),
                condition=data.get('condition', '')
            )
            
            return JsonResponse({
                'success': True,
                'data': result
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def valuation_detail(request, pk):
    """Detail view for a specific valuation"""
    try:
        valuation = ItemValuation.objects.get(pk=pk, user=request.user)
        return render(request, 'valuations/detail.html', {
            'valuation': valuation
        })
    except ItemValuation.DoesNotExist:
        messages.error(request, 'Valuation not found.')
        return redirect('valuations:list')
'''

# ===== DJANGO INTEGRATION EXAMPLES =====

def get_config(django_settings=None) -> AutoMarketConfig:
    """
    Get configuration object
    Can be used with or without Django settings
    """
    if django_settings:
        return AutoMarketConfig.from_django_settings(django_settings)
    return AutoMarketConfig()

def allowed_file(filename: str, config: AutoMarketConfig = None) -> bool:
    """Check if file extension is allowed"""
    if config is None:
        config = get_config()
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def clean_text(text: str) -> str:
    """Clean text for search optimization"""
    # Remove special characters but keep spaces and hyphens
    cleaned = re.sub(r'[^\w\s\-]', ' ', text)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# ===== DJANGO-COMPATIBLE SERVICE INTERFACES =====

class ImageAnalysisInterface(ABC):
    """Abstract interface for image analysis services"""
    
    @abstractmethod
    def analyze_condition(self, image_path: str) -> ItemCondition:
        """Analyze image condition"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Preprocess image for analysis"""
        pass

class PriceScrapingInterface(ABC):
    """Abstract interface for price scraping services"""
    
    @abstractmethod
    def scrape_prices(self, item_name: str, description: str, condition: str) -> List[float]:
        """Scrape prices from market sources"""
        pass

class ValuationInterface(ABC):
    """Abstract interface for valuation services"""
    
    @abstractmethod
    def calculate_valuation(self, prices: List[float], condition: str, 
                          image_analysis: List[ItemCondition], 
                          location: Optional[str] = None) -> PriceEstimate:
        """Calculate item valuation"""
        pass

# ===== DJANGO-COMPATIBLE SERVICE CLASSES =====

class AIModelManager:
    """
    Django-compatible AI model manager
    Handles loading and caching of AI models
    """
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: AutoMarketConfig = None):
        if not hasattr(self, 'initialized'):
            self.config = config or get_config()
            self.image_classifier = None
            self.image_processor = None
            self.damage_detector = None
            self.car_model_classifier = None
            self.initialized = True
    
    def load_models(self) -> bool:
        """
        Load AI models - can be called from Django management command
        Usage: python manage.py load_ai_models
        """
        if self._models_loaded:
            return True
            
        try:
            logger.info("Loading AI models...")
            
            # Load main image classifier
            self._load_main_classifier()
            
            # Load specialized models
            self._load_damage_detector()
            self._load_car_classifier()
            
            self._models_loaded = True
            logger.info("All AI models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            return False
    
    def _load_main_classifier(self):
        """Load main image classification model"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                try:
                    self.image_processor = AutoImageProcessor.from_pretrained(
                        self.config.DEFAULT_MODEL_NAME,
                        resume_download=False,
                        force_download=False
                    )
                except Exception:
                    self.image_processor = None
                
                try:
                    model = AutoModelForImageClassification.from_pretrained(
                        self.config.DEFAULT_MODEL_NAME,
                        resume_download=False,
                        force_download=False
                    )
                    
                    if self.image_processor:
                        self.image_classifier = pipeline(
                            "image-classification",
                            model=model,
                            image_processor=self.image_processor,
                            top_k=5
                        )
                    else:
                        self.image_classifier = pipeline(
                            "image-classification",
                            model=self.config.DEFAULT_MODEL_NAME,
                            top_k=5
                        )
                except Exception:
                    # Fallback model
                    self.image_classifier = pipeline(
                        "image-classification",
                        model=self.config.FALLBACK_MODEL_NAME,
                        top_k=5
                    )
                    
        except Exception as e:
            logger.error(f"Failed to load main classifier: {e}")
            self.image_classifier = None
    
    def _load_damage_detector(self):
        """Load damage detection model"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.damage_detector = pipeline(
                    "object-detection",
                    model=self.config.DAMAGE_DETECTOR_MODEL,
                    top_k=10
                )
        except Exception as e:
            logger.warning(f"Could not load damage detector: {e}")
            self.damage_detector = None
    
    def _load_car_classifier(self):
        """Load car-specific classifier"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.car_model_classifier = pipeline(
                    "image-classification",
                    model=self.config.FALLBACK_MODEL_NAME,
                    top_k=5
                )
        except Exception as e:
            logger.warning(f"Could not load car classifier: {e}")
            self.car_model_classifier = None
    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return self._models_loaded and self.image_classifier is not None

class ImageAnalysisService(ImageAnalysisInterface):
    """
    Django-compatible image analysis service
    Usage in Django views:
        service = ImageAnalysisService()
        result = service.analyze_condition(image_path)
    """
    
    def __init__(self, config: AutoMarketConfig = None):
        self.config = config or get_config()
        self.model_manager = AIModelManager(config)
        if not self.model_manager.is_ready():
            self.model_manager.load_models()
    
    def analyze_condition(self, image_path: str) -> ItemCondition:
        """Analyze image condition using multiple AI techniques"""
        try:
            # Preprocess image
            processed = self.preprocess_image(image_path)
            if not processed:
                return ItemCondition(
                    condition_score=0.5,
                    detected_condition='Fair',
                    defects_detected=[],
                    confidence=0.3
                )
            
            # Multi-modal analysis
            analysis_results = {
                'ml_classification': None,
                'damage_detection': None,
                'color_analysis': None,
                'texture_analysis': None,
                'edge_analysis': None
            }
            
            # ML Classification
            if self.model_manager.image_classifier:
                try:
                    results = self.model_manager.image_classifier(image_path)
                    analysis_results['ml_classification'] = results
                except Exception as e:
                    logger.error(f"ML classification error: {e}")
            
            # Damage Detection
            if self.model_manager.damage_detector:
                try:
                    damage_results = self.model_manager.damage_detector(image_path)
                    analysis_results['damage_detection'] = damage_results
                except Exception as e:
                    logger.error(f"Damage detection error: {e}")
            
            # Color, texture, and edge analysis
            analysis_results['color_analysis'] = self._analyze_color_quality(processed['image_rgb'])
            analysis_results['texture_analysis'] = self._analyze_texture_quality(processed['image_rgb'])
            analysis_results['edge_analysis'] = self._analyze_edge_quality(processed)
            
            # Combine analyses
            final_assessment = self._combine_condition_analyses(analysis_results)
            
            return ItemCondition(
                condition_score=final_assessment['condition_score'],
                detected_condition=final_assessment['detected_condition'],
                defects_detected=final_assessment['defects_detected'],
                confidence=final_assessment['confidence'],
                analysis_details=analysis_results
            )
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return ItemCondition(
                condition_score=0.5,
                detected_condition='Fair',
                defects_detected=[],
                confidence=0.3
            )
    
    def preprocess_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Preprocess image using OpenCV"""
        try:
            # Create temp directory if it doesn't exist
            os.makedirs(self.config.TEMP_IMAGES_FOLDER, exist_ok=True)
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Resize to standard size
            image = cv2.resize(image, (224, 224))
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic preprocessing for defect detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return {
                'image_rgb': image_rgb,
                'edge_density': edge_density,
                'size': image.shape
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

# ===== IMAGE ANALYSIS FUNCTIONS =====

            return None

    def _analyze_color_quality(self, image_rgb):
        """Analyze color quality to detect fading, discoloration"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # Calculate color statistics
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # Calculate metrics
            saturation_mean = np.mean(saturation)
            saturation_std = np.std(saturation)
            value_mean = np.mean(value)
            value_std = np.std(value)
            
            # Color uniformity (less uniform = more wear/damage)
            color_uniformity = 1.0 - (saturation_std / 255.0)
            
            # Brightness consistency
            brightness_consistency = 1.0 - (value_std / 255.0)
            
            # Overall color quality score
            color_quality = (color_uniformity + brightness_consistency) / 2.0
            
            return {
                'color_quality_score': color_quality,
                'saturation_mean': saturation_mean,
                'color_uniformity': color_uniformity,
                'brightness_consistency': brightness_consistency
            }
        except Exception as e:
            logger.error(f"Color analysis error: {e}")
            return {'color_quality_score': 0.5}

    def _analyze_texture_quality(self, image_rgb):
        """Analyze texture to detect scratches, wear patterns"""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Calculate Local Binary Pattern for texture analysis
            try:
                from skimage import feature
                # LBP parameters
                radius = 3
                n_points = 8 * radius
                lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
                
                # Calculate texture uniformity
                hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-7)
                
                # Texture uniformity score (higher = more uniform = better condition)
                texture_uniformity = 1.0 - np.sum(hist ** 2)
                
            except ImportError:
                # Fallback texture analysis using gradient variance
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                texture_uniformity = 1.0 - (np.std(gradient_magnitude) / np.mean(gradient_magnitude + 1e-7))
            
            # Surface roughness estimation
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            surface_smoothness = 1.0 / (1.0 + laplacian_var / 1000.0)  # Normalize
            
            return {
                'texture_uniformity': max(0, min(1, texture_uniformity)),
                'surface_smoothness': max(0, min(1, surface_smoothness)),
                'overall_texture_score': (texture_uniformity + surface_smoothness) / 2.0
            }
        except Exception as e:
            logger.error(f"Texture analysis error: {e}")
            return {'overall_texture_score': 0.5}

    def _analyze_edge_quality(self, processed_data):
        """Enhanced edge analysis for damage detection"""
        try:
            edge_density = processed_data['edge_density']
            
            # Additional edge-based metrics
            image_rgb = processed_data['image_rgb']
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Multi-scale edge detection
            edges_fine = cv2.Canny(gray, 50, 150)
            edges_coarse = cv2.Canny(gray, 100, 200)
            
            # Calculate edge distribution
            fine_edge_density = np.sum(edges_fine > 0) / (edges_fine.shape[0] * edges_fine.shape[1])
            coarse_edge_density = np.sum(edges_coarse > 0) / (edges_coarse.shape[0] * edges_coarse.shape[1])
            
            # Edge ratio (fine/coarse) indicates detail level
            edge_ratio = fine_edge_density / (coarse_edge_density + 1e-7)
            
            # Edge continuity (connected components analysis)
            num_labels, labels = cv2.connectedComponents(edges_fine)
            edge_fragmentation = num_labels / (fine_edge_density * edges_fine.size + 1e-7)
            
            return {
                'edge_density': edge_density,
                'fine_edge_density': fine_edge_density,
                'coarse_edge_density': coarse_edge_density,
                'edge_ratio': edge_ratio,
                'edge_fragmentation': min(1.0, edge_fragmentation),
                'edge_quality_score': 1.0 - min(1.0, edge_density + edge_fragmentation / 2.0)
            }
        except Exception as e:
            logger.error(f"Enhanced edge analysis error: {e}")
            return {'edge_quality_score': 0.5}

    def _combine_condition_analyses(self, analysis_results):
        """Combine multiple analysis results for final condition assessment"""
        try:
            # Initialize scores
            scores = []
            weights = []
            defects_detected = []
            
            # ML Classification (weight: 0.3)
            if analysis_results['ml_classification']:
                ml_score = self._get_ml_condition_score(analysis_results['ml_classification'])
                scores.append(ml_score)
                weights.append(0.3)
            
            # Color Analysis (weight: 0.2)
            if analysis_results['color_analysis']:
                color_score = analysis_results['color_analysis'].get('color_quality_score', 0.5)
                scores.append(color_score)
                weights.append(0.2)
                
                if color_score < 0.4:
                    defects_detected.append("Color fading or discoloration detected")
            
            # Texture Analysis (weight: 0.2)
            if analysis_results['texture_analysis']:
                texture_score = analysis_results['texture_analysis'].get('overall_texture_score', 0.5)
                scores.append(texture_score)
                weights.append(0.2)
                
                if texture_score < 0.4:
                    defects_detected.append("Surface wear or scratches detected")
            
            # Edge Analysis (weight: 0.15)
            if analysis_results['edge_analysis']:
                edge_score = analysis_results['edge_analysis'].get('edge_quality_score', 0.5)
                scores.append(edge_score)
                weights.append(0.15)
                
                if edge_score < 0.4:
                    defects_detected.append("High edge density indicating possible damage")
            
            # Damage Detection (weight: 0.15)
            if analysis_results['damage_detection']:
                damage_score = self._get_damage_detection_score(analysis_results['damage_detection'])
                scores.append(damage_score)
                weights.append(0.15)
                
                if damage_score < 0.5:
                    defects_detected.append("Damage detected by AI analysis")
            
            # Calculate weighted average
            if scores and weights:
                # Normalize weights
                total_weight = sum(weights)
                weights = [w/total_weight for w in weights]
                
                final_score = sum(score * weight for score, weight in zip(scores, weights))
            else:
                final_score = 0.5  # Default
            
            # Determine condition category
            if final_score >= 0.8:
                detected_condition = 'Excellent'
            elif final_score >= 0.65:
                detected_condition = 'Good'
            elif final_score >= 0.4:
                detected_condition = 'Fair'
            else:
                detected_condition = 'Poor'
            
            # Calculate confidence based on analysis completeness
            confidence = min(0.95, 0.4 + (len(scores) * 0.1))
            
            return {
                'condition_score': final_score,
                'detected_condition': detected_condition,
                'defects_detected': defects_detected,
                'confidence': confidence,
                'analysis_details': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error combining analyses: {e}")
            return {'condition_score': 0.5, 'defects_detected': [], 'confidence': 0.3, 'detected_condition': 'Fair'}

    def _get_ml_condition_score(self, ml_results):
        """Extract condition score from ML classification results"""
        try:
            if not ml_results:
                return 0.5
                
            top_prediction = ml_results[0]
            label = top_prediction['label'].lower()
            score = top_prediction['score']
            
            # Enhanced condition keyword mapping
            condition_keywords = {
                'excellent': ['new', 'pristine', 'mint', 'perfect', 'clean', 'spotless', 'flawless'],
                'good': ['good', 'functional', 'working', 'intact', 'solid', 'fine', 'decent'],
                'fair': ['used', 'worn', 'average', 'moderate', 'normal', 'okay', 'acceptable'],
                'poor': ['damaged', 'broken', 'cracked', 'scratched', 'dented', 'worn', 'defective', 'faulty']
            }
            
            condition_scores = {
                'excellent': 0.9,
                'good': 0.7,
                'fair': 0.5,
                'poor': 0.3
            }
            
            # Find matching condition
            detected_condition = 'fair'  # default
            for condition, keywords in condition_keywords.items():
                if any(keyword in label for keyword in keywords):
                    detected_condition = condition
                    break
            
            base_score = condition_scores[detected_condition]
            
            # Adjust based on ML confidence
            confidence_adjustment = (score - 0.5) * 0.2  # -0.1 to +0.1 adjustment
            final_score = max(0.1, min(0.95, base_score + confidence_adjustment))
            
            return final_score
        except Exception as e:
            logger.error(f"Error processing ML results: {e}")
            return 0.5

    def _get_damage_detection_score(self, damage_results):
        """Extract damage score from object detection results"""
        try:
            if not damage_results:
                return 0.8  # No damage detected
                
            # Count damage-related detections
            damage_keywords = ['scratch', 'dent', 'crack', 'hole', 'stain', 'wear', 'damage', 'defect']
            damage_count = 0
            total_confidence = 0
            
            for detection in damage_results:
                label = detection.get('label', '').lower()
                confidence = detection.get('score', 0)
                
                if any(keyword in label for keyword in damage_keywords):
                    damage_count += 1
                    total_confidence += confidence
            
            if damage_count == 0:
                return 0.8  # No damage detected
            
            # Calculate damage severity
            avg_damage_confidence = total_confidence / damage_count
            damage_severity = min(1.0, damage_count * avg_damage_confidence / 5.0)
            
            # Return inverted score (less damage = higher score)
            return max(0.1, 1.0 - damage_severity)
            
        except Exception as e:
            logger.error(f"Error processing damage detection: {e}")
            return 0.5

class PriceScrapingService(PriceScrapingInterface):
    """
    Django-compatible price scraping service
    Usage in Django views:
        service = PriceScrapingService()
        prices = service.scrape_prices(item_name, description, condition)
    """
    
    def __init__(self, config: AutoMarketConfig = None):
        self.config = config or get_config()
    
    def scrape_prices(self, item_name: str, description: str, condition: str, 
                     detected_condition: Optional[str] = None) -> List[float]:
        """Scrape prices from multiple sources"""
        all_prices = []
        
        # Get eBay prices
        ebay_prices = self._scrape_ebay_prices(item_name, description, condition, detected_condition)
        all_prices.extend(ebay_prices)
        
        # Additional sources can be added here
        # facebook_prices = self._scrape_facebook_marketplace(item_name, condition, detected_condition)
        # amazon_prices = self._scrape_amazon_used_prices(item_name, description, condition)
        
        return all_prices

    def _scrape_ebay_prices(self, item_name: str, description: str, condition: str, 
                           detected_condition: Optional[str] = None) -> List[float]:
        """Scrape eBay for sold listings to get market prices"""
        try:
            # Use AI-detected condition if available, otherwise use user-provided condition
            search_condition = detected_condition if detected_condition else condition
            
            # Build search query
            search_query = self._build_search_query(item_name, description, search_condition)
            encoded_query = quote_plus(search_query[:80])
            
            logger.info(f"eBay search query: '{search_query}' -> '{encoded_query}'")
            
            # eBay sold listings URL
            url = f"https://www.ebay.com/sch/i.html?_from=R40&_nkw={encoded_query}&_sacat=0&rt=nc&LH_Sold=1&LH_Complete=1"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(*self.config.DELAY_RANGE))
            
            response = requests.get(url, headers=headers, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            prices = self._extract_prices_from_soup(soup)
            
            logger.info(f"Found {len(prices)} prices for {item_name}")
            return prices[:self.config.MAX_PRICE_RESULTS]
            
        except Exception as e:
            logger.error(f"Error scraping eBay prices: {e}")
            return []

    def _build_search_query(self, item_name: str, description: str, condition: str) -> str:
        """Build optimized search query"""
        full_text = f"{item_name} {description}".lower()
        
        # Extract important keywords
        tech_keywords = re.findall(
            r'\b(?:intel|amd|nvidia|apple|samsung|lg|hp|dell|lenovo|asus|acer|msi|macbook|iphone|ipad|laptop|desktop|monitor|gb|tb|ghz|core|i3|i5|i7|i9|ryzen|graphics|ssd|hdd|ram|memory)\b', 
            full_text
        )
        
        # Extract numbers that might be important
        numbers = re.findall(r'\b\d+(?:gb|tb|ghz|inch|")\b', full_text)
        
        # Clean and combine search terms
        search_terms = []
        
        # Add cleaned item name
        clean_name = re.sub(r'[^\w\s-]', ' ', item_name).strip()
        if clean_name:
            search_terms.append(clean_name)
        
        # Enhanced condition mapping for USED items
        condition_mapping = {
            'Excellent': 'used excellent condition pre-owned',
            'Good': 'used good condition pre-owned working',
            'Fair': 'used fair condition pre-owned functional',
            'Poor': 'used poor condition for parts repair'
        }
        
        if condition in condition_mapping:
            search_terms.append(condition_mapping[condition])
        
        # Always add "used" keyword
        if 'used' not in ' '.join(search_terms).lower():
            search_terms.append('used')
        
        # Add tech keywords and numbers
        search_terms.extend(tech_keywords[:5])
        search_terms.extend(numbers[:3])
        
        search_query = ' '.join(search_terms)
        
        # Fallback if no good terms found
        if not search_query.strip():
            search_query = f"used {item_name[:50]} {condition.lower()}"
        
        return search_query

    def _extract_prices_from_soup(self, soup) -> List[float]:
        """Extract prices from BeautifulSoup object"""
        prices = []
        
        # Try different price selectors
        price_selectors = [
            '.s-item__price .notranslate',
            '.s-item__price',
            '.u-flL.condText', 
            '.s-item__detail--primary .s-item__price',
            '.notranslate',
            'span[class*="price"]',
            'div[class*="price"]',
            '.s-item__price-range'
        ]
        
        for selector in price_selectors:
            price_elements = soup.select(selector)
            for element in price_elements:
                try:
                    price_text = element.get_text().strip()
                    # Extract numeric price
                    price_matches = re.findall(r'\$([0-9,]+\.?[0-9]*)', price_text)
                    for price_match in price_matches:
                        try:
                            price = float(price_match.replace(',', ''))
                            if 1 <= price <= 10000:  # Reasonable price range
                                prices.append(price)
                        except ValueError:
                            continue
                except (ValueError, AttributeError):
                    continue
            
            if len(prices) >= 15:
                break
        
        # Fallback: search entire page text
        if not prices:
            all_text = soup.get_text()
            price_matches = re.findall(r'\$([0-9,]+\.?[0-9]*)', all_text)
            for match in price_matches[:30]:
                try:
                    price = float(match.replace(',', ''))
                    if 5 <= price <= 10000:
                        prices.append(price)
                except ValueError:
                    continue
        
        return self._filter_prices(prices)

    def _filter_prices(self, prices: List[float]) -> List[float]:
        """Filter and clean price data"""
        if not prices:
            return []
        
        prices = list(set(prices))  # Remove duplicates
        prices.sort()
        
        if len(prices) >= 3:
            # Remove statistical outliers
            prices_array = np.array(prices)
            mean_price = np.mean(prices_array)
            std_price = np.std(prices_array)
            
            if std_price > 0:
                # Keep prices within 2 standard deviations
                filtered_prices = [p for p in prices if abs(p - mean_price) <= 2 * std_price]
                if len(filtered_prices) >= 3:
                    prices = filtered_prices
        
        return prices

# ===== MAIN DJANGO SERVICE CLASS =====

class ItemValuationService:
    """
    Main Django-compatible service for item valuation
    
    Usage in Django views:
        from auto_market_core import ItemValuationService
        
        service = ItemValuationService()
        result = service.estimate_value(
            item_name="MacBook Pro",
            description="2019 model with 8GB RAM",
            condition="Good",
            issues="Minor scratches",
            image_paths=image_paths,
            user_location="San Francisco"
        )
    """
    
    def __init__(self, config: AutoMarketConfig = None):
        self.config = config or get_config()
        self.image_service = ImageAnalysisService(config)
        self.price_service = PriceScrapingService(config)
    
    def estimate_value(self, item_name: str, description: str, condition: str,
                      issues: str = "", image_paths: Optional[List[str]] = None,
                      user_location: Optional[str] = None) -> PriceEstimate:
        """
        Main function to estimate item value
        
        Args:
            item_name: Name of the item
            description: Description of the item  
            condition: Condition ('Excellent', 'Good', 'Fair', 'Poor')
            issues: Any issues or defects
            image_paths: List of paths to images for analysis
            user_location: User's location for regional pricing
            
        Returns:
            PriceEstimate object with valuation results
        """
        logger.info(f"Starting valuation for: {item_name} ({condition})")
        
        # Initialize
        image_analysis_results = []
        
        # Analyze images if provided
        if image_paths:
            for image_path in image_paths:
                try:
                    if os.path.exists(image_path):
                        analysis = self.image_service.analyze_condition(image_path)
                        image_analysis_results.append(analysis)
                        logger.info(f"Analyzed image: {image_path}")
                except Exception as e:
                    logger.error(f"Error analyzing image {image_path}: {e}")
        
        # Get AI-detected condition
        ai_detected_condition = condition
        if image_analysis_results:
            best_analysis = max(image_analysis_results, key=lambda x: x.confidence)
            ai_detected_condition = best_analysis.detected_condition
            logger.info(f"Using AI-detected condition: {ai_detected_condition}")
        
        # Scrape market data
        logger.info("Scraping market data...")
        scraped_prices = self.price_service.scrape_prices(
            item_name, description, condition, ai_detected_condition
        )
        
        # Calculate valuation
        valuation_data = self._calculate_valuation(
            scraped_prices, condition, issues, image_analysis_results, 
            user_location, item_name, ai_detected_condition
        )
        
        # Create PriceEstimate object
        estimate = PriceEstimate(**valuation_data)
        
        logger.info(f"Valuation complete: ${estimate.estimated_value} ({estimate.confidence} confidence)")
        
        return estimate

    def _calculate_valuation(self, prices: List[float], condition: str, issues: str,
                           image_analysis_results: List[ItemCondition], 
                           user_location: Optional[str], item_name: str,
                           ai_detected_condition: str) -> Dict[str, Any]:
        """Calculate valuation with all factors considered"""
        try:
            # Get condition confidence from image analysis
            condition_confidence = 0.5
            if image_analysis_results:
                best_analysis = max(image_analysis_results, key=lambda x: x.confidence)
                condition_confidence = best_analysis.confidence
            
            # Regional price adjustments
            regional_multiplier = self._get_regional_price_multiplier(user_location)
            
            if not prices or len(prices) < 3:
                # Fallback pricing
                base_estimate = self._get_enhanced_fallback_estimate(
                    item_name, ai_detected_condition, condition_confidence
                )
                adjusted_estimate = int(base_estimate * regional_multiplier)
                
                confidence_level = 'Low'
                if condition_confidence > 0.7:
                    confidence_level = 'Medium'
                
                return {
                    'estimated_value': adjusted_estimate,
                    'range_low': int(adjusted_estimate * 0.7),
                    'range_high': int(adjusted_estimate * 1.3),
                    'confidence': confidence_level,
                    'ai_detected_condition': ai_detected_condition,
                    'condition_confidence': condition_confidence,
                    'market_data_points': len(prices),
                    'regional_multiplier': regional_multiplier,
                    'timestamp': datetime.now().isoformat(),
                    'note': f'AI detected condition: {ai_detected_condition}. Regional adjustment: {regional_multiplier:.2f}x. Estimate based on enhanced market analysis (limited data)'
                }
            
            # Market data analysis
            market_analysis = self._analyze_market_trends(prices, ai_detected_condition)
            base_estimate = market_analysis['recommended_price']
            final_estimate = int(base_estimate * regional_multiplier)
            
            # Calculate confidence
            confidence = self._calculate_confidence_score(
                len(prices), condition_confidence, market_analysis['price_volatility']
            )
            
            return {
                'estimated_value': final_estimate,
                'range_low': market_analysis['confidence_range']['low'],
                'range_high': market_analysis['confidence_range']['high'],
                'confidence': confidence,
                'ai_detected_condition': ai_detected_condition,
                'condition_confidence': condition_confidence,
                'market_data_points': len(prices),
                'regional_multiplier': regional_multiplier,
                'market_analysis': market_analysis,
                'timestamp': datetime.now().isoformat(),
                'note': f'Enhanced market analysis with {len(prices)} data points. Regional adjustment: {regional_multiplier:.2f}x. Condition: {ai_detected_condition}'
            }
            
        except Exception as e:
            logger.error(f"Error in valuation calculation: {e}")
            return {
                'estimated_value': 50,
                'range_low': 30,
                'range_high': 80,
                'confidence': 'Low',
                'ai_detected_condition': condition,
                'condition_confidence': 0.3,
                'market_data_points': 0,
                'regional_multiplier': 1.0,
                'timestamp': datetime.now().isoformat(),
                'note': 'Error occurred during valuation calculation'
            }

    def _get_regional_price_multiplier(self, user_location: Optional[str]) -> float:
        """Get regional price adjustment multiplier"""
        if not user_location:
            return 1.0
        
        location_key = user_location.lower().replace(' ', '_').replace(',', '')
        return self.config.REGIONAL_MULTIPLIERS.get(location_key, 1.0)

    def _get_enhanced_fallback_estimate(self, item_name: str, condition: str, confidence: float) -> int:
        """Enhanced fallback estimates with more categories"""
        item_name_lower = item_name.lower()
        
        # Enhanced estimates by category
        enhanced_estimates = {
            'laptop': {'Excellent': 350, 'Good': 250, 'Fair': 180, 'Poor': 100},
            'macbook': {'Excellent': 600, 'Good': 450, 'Fair': 320, 'Poor': 180},
            'iphone': {'Excellent': 300, 'Good': 220, 'Fair': 150, 'Poor': 80},
            'ipad': {'Excellent': 250, 'Good': 180, 'Fair': 130, 'Poor': 70},
            'phone': {'Excellent': 150, 'Good': 100, 'Fair': 70, 'Poor': 35},
            'tablet': {'Excellent': 120, 'Good': 80, 'Fair': 55, 'Poor': 30},
            'monitor': {'Excellent': 150, 'Good': 110, 'Fair': 75, 'Poor': 40},
            'camera': {'Excellent': 220, 'Good': 160, 'Fair': 110, 'Poor': 60},
            'car': {'Excellent': 12000, 'Good': 8500, 'Fair': 6000, 'Poor': 3500},
            'default': {'Excellent': 100, 'Good': 70, 'Fair': 45, 'Poor': 25}
        }
        
        # Find best matching category
        best_match = 'default'
        for category in enhanced_estimates.keys():
            if category in item_name_lower:
                best_match = category
                break
        
        estimates = enhanced_estimates[best_match]
        base_price = estimates.get(condition, estimates['Fair'])
        
        # Adjust based on confidence
        confidence_multiplier = 0.85 + (confidence * 0.3)
        return int(base_price * confidence_multiplier)

    def _analyze_market_trends(self, prices: List[float], condition: str) -> Dict[str, Any]:
        """Analyze market trends and provide pricing recommendations"""
        try:
            prices_array = np.array(prices)
            
            # Basic statistics
            median_price = np.median(prices_array)
            std_price = np.std(prices_array)
            mean_price = np.mean(prices_array)
            
            # Condition-based pricing strategy
            condition_multipliers = {
                'Excellent': 0.85,
                'Good': 0.70,
                'Fair': 0.55,
                'Poor': 0.35
            }
            
            multiplier = condition_multipliers.get(condition, 0.55)
            recommended_price = median_price * multiplier
            
            # Market volatility assessment
            cv = std_price / mean_price if mean_price > 0 else 1.0
            if cv < 0.2:
                volatility = 'Low'
                range_factor = 0.15
            elif cv < 0.4:
                volatility = 'Medium'
                range_factor = 0.25
            else:
                volatility = 'High'
                range_factor = 0.35
            
            # Calculate confidence range
            confidence_range = {
                'low': int(recommended_price * (1 - range_factor)),
                'high': int(recommended_price * (1 + range_factor))
            }
            
            return {
                'recommended_price': int(recommended_price),
                'confidence_range': confidence_range,
                'price_volatility': volatility,
                'market_median': int(median_price),
                'condition_multiplier': multiplier
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {e}")
            return {
                'recommended_price': 50,
                'confidence_range': {'low': 35, 'high': 70},
                'price_volatility': 'High'
            }

    def _calculate_confidence_score(self, data_points: int, condition_confidence: float, 
                                  volatility: str) -> str:
        """Calculate overall confidence score"""
        # Data quantity score
        if data_points >= self.config.MIN_DATA_POINTS_HIGH_CONFIDENCE:
            quantity_score = 1.0
        elif data_points >= self.config.MIN_DATA_POINTS_MEDIUM_CONFIDENCE:
            quantity_score = 0.8
        elif data_points >= self.config.MIN_DATA_POINTS_LOW_CONFIDENCE:
            quantity_score = 0.6
        else:
            quantity_score = 0.4
        
        # Volatility score
        volatility_scores = {'Low': 1.0, 'Medium': 0.7, 'High': 0.4}
        volatility_score = volatility_scores.get(volatility, 0.5)
        
        # Combined confidence
        overall_confidence = (
            quantity_score * 0.5 +
            condition_confidence * 0.3 +
            volatility_score * 0.2
        )
        
        # Convert to categorical
        if overall_confidence >= 0.8:
            return 'High'
        elif overall_confidence >= 0.6:
            return 'Medium'
        else:
            return 'Low'

# ===== DJANGO INTEGRATION HELPERS =====

class DjangoValuationView:
    """
    Example Django view integration helper
    
    Usage in Django views.py:
        from auto_market_core import DjangoValuationView
        
        class ItemValuationAPIView(APIView):
            def post(self, request):
                return DjangoValuationView.handle_valuation_request(request)
    """
    
    @staticmethod
    def handle_valuation_request(request):
        """
        Handle Django request for item valuation
        
        Expected POST data:
        - itemName: string
        - description: string  
        - condition: string
        - issues: string (optional)
        - userLocation: string (optional)
        - photos: files (optional)
        
        Returns Django JsonResponse
        """
        try:
            from django.http import JsonResponse
            from django.core.files.storage import default_storage
            from django.core.files.base import ContentFile
            
            # Extract data from request
            valuation_request = ValuationRequest.from_django_request(request)
            
            # Validate required fields
            if not all([valuation_request.item_name, valuation_request.description, 
                       valuation_request.condition]):
                return JsonResponse({
                    'error': 'Missing required fields',
                    'estimated_value': 0,
                    'range_low': 0,
                    'range_high': 0,
                    'confidence': 'Low'
                }, status=400)
            
            # Process uploaded images
            image_paths = []
            uploaded_files = request.FILES.getlist('photos')
            
            for file in uploaded_files:
                if allowed_file(file.name):
                    try:
                        # Save file temporarily
                        filename = f"temp_{int(time.time())}_{file.name}"
                        file_path = default_storage.save(filename, ContentFile(file.read()))
                        image_paths.append(default_storage.path(file_path))
                    except Exception as e:
                        logger.error(f"Error processing uploaded file: {e}")
            
            # Perform valuation
            service = ItemValuationService()
            result = service.estimate_value(
                item_name=valuation_request.item_name,
                description=valuation_request.description,
                condition=valuation_request.condition,
                issues=valuation_request.issues,
                image_paths=image_paths,
                user_location=valuation_request.user_location
            )
            
            # Clean up temporary files
            for file_path in image_paths:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
            
            return JsonResponse(result.to_dict())
            
        except Exception as e:
            logger.error(f"Error in Django valuation request: {e}")
            return JsonResponse({
                'error': 'Internal server error occurred',
                'estimated_value': 0,
                'range_low': 0,
                'range_high': 0,
                'confidence': 'Low'
            }, status=500)

class DjangoManagementCommands:
    """
    Example Django management command helpers
    
    Create management/commands/load_ai_models.py:
        from auto_market_core import DjangoManagementCommands
        
        class Command(BaseCommand):
            def handle(self, *args, **options):
                DjangoManagementCommands.load_ai_models_command(self)
    """
    
    @staticmethod
    def load_ai_models_command(command_instance):
        """Load AI models management command"""
        command_instance.stdout.write("Loading AI models...")
        
        model_manager = AIModelManager()
        success = model_manager.load_models()
        
        if success:
            command_instance.stdout.write(
                command_instance.style.SUCCESS('Successfully loaded AI models')
            )
        else:
            command_instance.stdout.write(
                command_instance.style.ERROR('Failed to load AI models')
            )

# ===== DJANGO MODELS EXAMPLE =====

def get_django_model_example():
    """
    Example Django model definition for storing valuation results
    
    Add this to your Django models.py:
    """
    model_code = '''
from django.db import models
from django.contrib.auth.models import User

class ItemValuation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    item_name = models.CharField(max_length=255)
    description = models.TextField()
    condition = models.CharField(max_length=50)
    issues = models.TextField(blank=True)
    user_location = models.CharField(max_length=100, blank=True)
    
    # Valuation results
    estimated_value = models.IntegerField()
    range_low = models.IntegerField()
    range_high = models.IntegerField()
    confidence = models.CharField(max_length=50)
    ai_detected_condition = models.CharField(max_length=50)
    condition_confidence = models.FloatField()
    market_data_points = models.IntegerField()
    regional_multiplier = models.FloatField()
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.item_name} - ${self.estimated_value}"

class ValuationImage(models.Model):
    valuation = models.ForeignKey(ItemValuation, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='valuation_images/')
    condition_score = models.FloatField(null=True, blank=True)
    detected_condition = models.CharField(max_length=50, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    analyzed_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Image for {self.valuation.item_name}"
'''
    return model_code

# ===== DJANGO SETTINGS EXAMPLE =====

def get_django_settings_example():
    """
    Example Django settings configuration
    
    Add this to your Django settings.py:
    """
    settings_code = '''
# Auto Market AI Configuration
AUTO_MARKET_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
AUTO_MARKET_TEMP_FOLDER = 'temp_images'
AUTO_MARKET_MODEL_NAME = 'google/efficientnet-b0'

# Regional pricing overrides
AUTO_MARKET_REGIONAL_MULTIPLIERS = {
    'custom_city': 1.25,
    'custom_region': 1.10,
}

# Media files configuration for image uploads
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Add to INSTALLED_APPS if creating a separate app
INSTALLED_APPS = [
    # ... other apps
    'auto_market',  # Your auto market app
]

# Celery configuration for async AI model loading (optional)
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
'''
    return settings_code

# ===== EXAMPLE USAGE =====

def django_example_usage():
    """
    Example of how to use the Django-compatible services
    """
    
    print("Django Auto Market AI Integration Example")
    print("=" * 50)
    
    # 1. Configuration
    print("\\n1. Configuration Setup:")
    config = AutoMarketConfig()
    print(f"   Default model: {config.DEFAULT_MODEL_NAME}")
    print(f"   Allowed extensions: {config.ALLOWED_EXTENSIONS}")
    
    # 2. Service initialization
    print("\\n2. Service Initialization:")
    service = ItemValuationService(config)
    print("   ItemValuationService initialized")
    
    # 3. Example valuation
    print("\\n3. Example Valuation:")
    result = service.estimate_value(
        item_name="MacBook Pro 13 inch",
        description="2019 MacBook Pro with 8GB RAM and 256GB SSD",
        condition="Good",
        issues="Minor scratches on lid",
        user_location="San Francisco"
    )
    
    print(f"   Estimated Value: ${result.estimated_value}")
    print(f"   Price Range: ${result.range_low} - ${result.range_high}")
    print(f"   Confidence: {result.confidence}")
    print(f"   AI Detected Condition: {result.ai_detected_condition}")
    
    # 4. Django integration examples
    print("\\n4. Django Integration:")
    print("   - Use DjangoValuationView.handle_valuation_request() in your views")
    print("   - Add management command with DjangoManagementCommands.load_ai_models_command()")
    print("   - Use the provided model examples for database storage")
    print("   - Configure settings with the provided examples")

if __name__ == "__main__":
    django_example_usage()
    """Load multiple specialized image classification models"""
    global image_classifier, image_processor, damage_detector, car_model_classifier
    try:
        # Load general EfficientNet-B0 for object classification
        model_name = "google/efficientnet-b0"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Try to load the image processor first
            try:
                image_processor = AutoImageProcessor.from_pretrained(
                    model_name, 
                    resume_download=False,
                    force_download=False
                )
            except Exception as e:
                logger.warning(f"Could not load AutoImageProcessor for {model_name}, using pipeline default: {e}")
                image_processor = None
            
            # Load the model
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    resume_download=False,
                    force_download=False
                )
                
                # Create pipeline with or without explicit image processor
                if image_processor:
                    image_classifier = pipeline(
                        "image-classification", 
                        model=model, 
                        image_processor=image_processor,
                        top_k=5
                    )
                else:
                    # Let the pipeline auto-detect the image processor
                    image_classifier = pipeline(
                        "image-classification", 
                        model=model_name,
                        top_k=5
                    )
            except Exception as e:
                logger.warning(f"Could not load {model_name}, trying alternative model: {e}")
                # Fallback to a more reliable model
                try:
                    image_classifier = pipeline(
                        "image-classification",
                        model="microsoft/resnet-50",
                        top_k=5
                    )
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {e2}")
                    image_classifier = None
        
        # Load specialized damage detection model (if available)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                damage_detector = pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",  # For detecting damage/defects
                    top_k=10
                )
        except Exception as e:
            logger.warning(f"Could not load damage detector: {e}")
            damage_detector = None
            
        # Load car-specific model for automotive items
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                car_model_classifier = pipeline(
                    "image-classification",
                    model="microsoft/resnet-50",  # Better for car recognition
                    top_k=5
                )
        except Exception as e:
            logger.warning(f"Could not load car model classifier: {e}")
            car_model_classifier = None
            
        logger.info("Enhanced image classifiers loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load image classifier: {e}")
        image_classifier = None
        return False

def preprocess_image(image_path):
    """Preprocess image using OpenCV"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Resize to standard size
        image = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Basic preprocessing for defect detection
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density (indicator of damage/wear)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return {
            'image_rgb': image_rgb,
            'edge_density': edge_density,
            'size': image.shape
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def analyze_color_quality(image_rgb):
    """Analyze color quality to detect fading, discoloration"""
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Calculate metrics
        saturation_mean = np.mean(saturation)
        saturation_std = np.std(saturation)
        value_mean = np.mean(value)
        value_std = np.std(value)
        
        # Color uniformity (less uniform = more wear/damage)
        color_uniformity = 1.0 - (saturation_std / 255.0)
        
        # Brightness consistency
        brightness_consistency = 1.0 - (value_std / 255.0)
        
        # Overall color quality score
        color_quality = (color_uniformity + brightness_consistency) / 2.0
        
        return {
            'color_quality_score': color_quality,
            'saturation_mean': saturation_mean,
            'color_uniformity': color_uniformity,
            'brightness_consistency': brightness_consistency
        }
    except Exception as e:
        logger.error(f"Color analysis error: {e}")
        return {'color_quality_score': 0.5}

def analyze_texture_quality(image_rgb):
    """Analyze texture to detect scratches, wear patterns"""
    try:
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Calculate Local Binary Pattern for texture analysis
        try:
            from skimage import feature
            # LBP parameters
            radius = 3
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate texture uniformity
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Texture uniformity score (higher = more uniform = better condition)
            texture_uniformity = 1.0 - np.sum(hist ** 2)
            
        except ImportError:
            # Fallback texture analysis using gradient variance
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            texture_uniformity = 1.0 - (np.std(gradient_magnitude) / np.mean(gradient_magnitude + 1e-7))
        
        # Surface roughness estimation
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        surface_smoothness = 1.0 / (1.0 + laplacian_var / 1000.0)  # Normalize
        
        return {
            'texture_uniformity': max(0, min(1, texture_uniformity)),
            'surface_smoothness': max(0, min(1, surface_smoothness)),
            'overall_texture_score': (texture_uniformity + surface_smoothness) / 2.0
        }
    except Exception as e:
        logger.error(f"Texture analysis error: {e}")
        return {'overall_texture_score': 0.5}

def analyze_edge_quality(processed_data):
    """Enhanced edge analysis for damage detection"""
    try:
        edge_density = processed_data['edge_density']
        
        # Additional edge-based metrics
        image_rgb = processed_data['image_rgb']
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 100, 200)
        
        # Calculate edge distribution
        fine_edge_density = np.sum(edges_fine > 0) / (edges_fine.shape[0] * edges_fine.shape[1])
        coarse_edge_density = np.sum(edges_coarse > 0) / (edges_coarse.shape[0] * edges_coarse.shape[1])
        
        # Edge ratio (fine/coarse) indicates detail level
        edge_ratio = fine_edge_density / (coarse_edge_density + 1e-7)
        
        # Edge continuity (connected components analysis)
        num_labels, labels = cv2.connectedComponents(edges_fine)
        edge_fragmentation = num_labels / (fine_edge_density * edges_fine.size + 1e-7)
        
        return {
            'edge_density': edge_density,
            'fine_edge_density': fine_edge_density,
            'coarse_edge_density': coarse_edge_density,
            'edge_ratio': edge_ratio,
            'edge_fragmentation': min(1.0, edge_fragmentation),
            'edge_quality_score': 1.0 - min(1.0, edge_density + edge_fragmentation / 2.0)
        }
    except Exception as e:
        logger.error(f"Enhanced edge analysis error: {e}")
        return {'edge_quality_score': 0.5}

def get_ml_condition_score(ml_results):
    """Extract condition score from ML classification results"""
    try:
        if not ml_results:
            return 0.5
            
        top_prediction = ml_results[0]
        label = top_prediction['label'].lower()
        score = top_prediction['score']
        
        # Enhanced condition keyword mapping
        condition_keywords = {
            'excellent': ['new', 'pristine', 'mint', 'perfect', 'clean', 'spotless', 'flawless'],
            'good': ['good', 'functional', 'working', 'intact', 'solid', 'fine', 'decent'],
            'fair': ['used', 'worn', 'average', 'moderate', 'normal', 'okay', 'acceptable'],
            'poor': ['damaged', 'broken', 'cracked', 'scratched', 'dented', 'worn', 'defective', 'faulty']
        }
        
        condition_scores = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
        
        # Find matching condition
        detected_condition = 'fair'  # default
        for condition, keywords in condition_keywords.items():
            if any(keyword in label for keyword in keywords):
                detected_condition = condition
                break
        
        base_score = condition_scores[detected_condition]
        
        # Adjust based on ML confidence
        confidence_adjustment = (score - 0.5) * 0.2  # -0.1 to +0.1 adjustment
        final_score = max(0.1, min(0.95, base_score + confidence_adjustment))
        
        return final_score
    except Exception as e:
        logger.error(f"Error processing ML results: {e}")
        return 0.5

def get_damage_detection_score(damage_results):
    """Extract damage score from object detection results"""
    try:
        if not damage_results:
            return 0.8  # No damage detected
            
        # Count damage-related detections
        damage_keywords = ['scratch', 'dent', 'crack', 'hole', 'stain', 'wear', 'damage', 'defect']
        damage_count = 0
        total_confidence = 0
        
        for detection in damage_results:
            label = detection.get('label', '').lower()
            confidence = detection.get('score', 0)
            
            if any(keyword in label for keyword in damage_keywords):
                damage_count += 1
                total_confidence += confidence
        
        if damage_count == 0:
            return 0.8  # No damage detected
        
        # Calculate damage severity
        avg_damage_confidence = total_confidence / damage_count
        damage_severity = min(1.0, damage_count * avg_damage_confidence / 5.0)
        
        # Return inverted score (less damage = higher score)
        return max(0.1, 1.0 - damage_severity)
        
    except Exception as e:
        logger.error(f"Error processing damage detection: {e}")
        return 0.5

def combine_condition_analyses(analysis_results):
    """Combine multiple analysis results for final condition assessment"""
    try:
        # Initialize scores
        scores = []
        weights = []
        defects_detected = []
        
        # ML Classification (weight: 0.3)
        if analysis_results['ml_classification']:
            ml_score = get_ml_condition_score(analysis_results['ml_classification'])
            scores.append(ml_score)
            weights.append(0.3)
        
        # Color Analysis (weight: 0.2)
        if analysis_results['color_analysis']:
            color_score = analysis_results['color_analysis'].get('color_quality_score', 0.5)
            scores.append(color_score)
            weights.append(0.2)
            
            if color_score < 0.4:
                defects_detected.append("Color fading or discoloration detected")
        
        # Texture Analysis (weight: 0.2)
        if analysis_results['texture_analysis']:
            texture_score = analysis_results['texture_analysis'].get('overall_texture_score', 0.5)
            scores.append(texture_score)
            weights.append(0.2)
            
            if texture_score < 0.4:
                defects_detected.append("Surface wear or scratches detected")
        
        # Edge Analysis (weight: 0.15)
        if analysis_results['edge_analysis']:
            edge_score = analysis_results['edge_analysis'].get('edge_quality_score', 0.5)
            scores.append(edge_score)
            weights.append(0.15)
            
            if edge_score < 0.4:
                defects_detected.append("High edge density indicating possible damage")
        
        # Damage Detection (weight: 0.15)
        if analysis_results['damage_detection']:
            damage_score = get_damage_detection_score(analysis_results['damage_detection'])
            scores.append(damage_score)
            weights.append(0.15)
            
            if damage_score < 0.5:
                defects_detected.append("Damage detected by AI analysis")
        
        # Calculate weighted average
        if scores and weights:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            
            final_score = sum(score * weight for score, weight in zip(scores, weights))
        else:
            final_score = 0.5  # Default
        
        # Determine condition category
        if final_score >= 0.8:
            detected_condition = 'Excellent'
        elif final_score >= 0.65:
            detected_condition = 'Good'
        elif final_score >= 0.4:
            detected_condition = 'Fair'
        else:
            detected_condition = 'Poor'
        
        # Calculate confidence based on analysis completeness
        confidence = min(0.95, 0.4 + (len(scores) * 0.1))
        
        return {
            'condition_score': final_score,
            'detected_condition': detected_condition,
            'defects_detected': defects_detected,
            'confidence': confidence,
            'analysis_details': analysis_results
        }
        
    except Exception as e:
        logger.error(f"Error combining analyses: {e}")
        return {'condition_score': 0.5, 'defects_detected': [], 'confidence': 0.3, 'detected_condition': 'Fair'}

def analyze_image_condition(image_path):
    """Enhanced image condition analysis with multiple techniques"""
    try:
        # Preprocess image
        processed = preprocess_image(image_path)
        if not processed:
            return {'condition_score': 0.5, 'defects_detected': [], 'confidence': 0.3, 'detected_condition': 'Fair'}
        
        # Multi-modal analysis
        analysis_results = {
            'ml_classification': None,
            'damage_detection': None,
            'color_analysis': None,
            'texture_analysis': None,
            'edge_analysis': None
        }
        
        # 1. ML Classification Analysis
        if image_classifier:
            try:
                results = image_classifier(image_path)
                analysis_results['ml_classification'] = results
            except Exception as e:
                logger.error(f"ML classification error: {e}")
        
        # 2. Damage Detection Analysis
        if 'damage_detector' in globals() and damage_detector:
            try:
                damage_results = damage_detector(image_path)
                analysis_results['damage_detection'] = damage_results
            except Exception as e:
                logger.error(f"Damage detection error: {e}")
        
        # 3. Color Analysis for wear/fading detection
        analysis_results['color_analysis'] = analyze_color_quality(processed['image_rgb'])
        
        # 4. Texture Analysis for surface condition
        analysis_results['texture_analysis'] = analyze_texture_quality(processed['image_rgb'])
        
        # 5. Enhanced Edge Analysis
        analysis_results['edge_analysis'] = analyze_edge_quality(processed)
        
        # 6. Combine all analyses for final condition assessment
        final_assessment = combine_condition_analyses(analysis_results)
        
        logger.info(f"Enhanced condition detected: {final_assessment['detected_condition']} (confidence: {final_assessment['confidence']:.2f})")
        
        return final_assessment
        
    except Exception as e:
        logger.error(f"Error in enhanced image analysis: {e}")
        return {'condition_score': 0.5, 'defects_detected': [], 'confidence': 0.3, 'detected_condition': 'Fair'}

# ===== PRICE SCRAPING FUNCTIONS =====

def scrape_ebay_prices(item_name, description, condition, detected_condition=None):
    """Scrape eBay for sold listings to get USED market prices based on detected condition"""
    try:
        # Use AI-detected condition if available, otherwise use user-provided condition
        search_condition = detected_condition if detected_condition else condition
        
        # Improve search query construction for USED items
        # Combine item name and description
        full_text = f"{item_name} {description}".lower()
        
        # Extract important keywords (brands, models, specs)
        tech_keywords = re.findall(r'\\b(?:intel|amd|nvidia|apple|samsung|lg|hp|dell|lenovo|asus|acer|msi|macbook|iphone|ipad|laptop|desktop|monitor|gb|tb|ghz|core|i3|i5|i7|i9|ryzen|graphics|ssd|hdd|ram|memory)\\b', full_text)
        
        # Extract numbers that might be important (model numbers, storage, etc.)
        numbers = re.findall(r'\\b\\d+(?:gb|tb|ghz|inch|")\\b', full_text)
        
        # Clean and combine search terms
        search_terms = []
        
        # Add cleaned item name
        clean_name = re.sub(r'[^\\w\\s-]', ' ', item_name).strip()
        if clean_name:
            search_terms.append(clean_name)
        
        # Enhanced condition mapping specifically for USED items
        condition_mapping = {
            'Excellent': 'used excellent condition pre-owned',
            'Good': 'used good condition pre-owned working',
            'Fair': 'used fair condition pre-owned functional',
            'Poor': 'used poor condition for parts repair'
        }
        
        if search_condition in condition_mapping:
            search_terms.append(condition_mapping[search_condition])
        
        # Always add "used" keyword to ensure we get used product prices
        if 'used' not in ' '.join(search_terms).lower():
            search_terms.append('used')
        
        # Add tech keywords
        search_terms.extend(tech_keywords[:5])  # Limit to 5 most relevant
        
        # Add important numbers
        search_terms.extend(numbers[:3])  # Limit to 3 numbers
        
        # Create search query
        search_query = ' '.join(search_terms)
        
        # Fallback if no good terms found
        if not search_query.strip():
            search_query = f"used {item_name[:50]} {search_condition.lower()}"  # Always include "used"
        
        # Encode for URL
        encoded_query = quote_plus(search_query[:80])  # Limit to 80 chars
        
        logger.info(f"USED items search query with AI condition '{search_condition}': '{search_query}' -> '{encoded_query}'")
        
        # eBay sold listings URL
        url = f"https://www.ebay.com/sch/i.html?_from=R40&_nkw={encoded_query}&_sacat=0&rt=nc&LH_Sold=1&LH_Complete=1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Add random delay to avoid being blocked
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find price elements with improved selectors
        prices = []
        
        # Try different price selectors (updated for current eBay structure)
        price_selectors = [
            '.s-item__price .notranslate',
            '.s-item__price',
            '.u-flL.condText', 
            '.s-item__detail--primary .s-item__price',
            '.notranslate',
            'span[class*="price"]',
            'div[class*="price"]',
            '.s-item__price-range'
        ]
        
        for selector in price_selectors:
            price_elements = soup.select(selector)
            for element in price_elements:
                try:
                    price_text = element.get_text().strip()
                    # Extract numeric price with improved regex
                    # Handle different price formats: $123, $123.45, $1,234.56
                    price_matches = re.findall(r'\\$([0-9,]+\\.?[0-9]*)', price_text)
                    for price_match in price_matches:
                        try:
                            price = float(price_match.replace(',', ''))
                            if 1 <= price <= 10000:  # Reasonable price range for most items
                                prices.append(price)
                        except ValueError:
                            continue
                except (ValueError, AttributeError):
                    continue
            
            if len(prices) >= 15:  # Collect more prices for better accuracy
                break
        
        # If still no prices found, try a more aggressive text search
        if not prices:
            # Look for any element containing price patterns in the entire page
            all_text = soup.get_text()
            # More comprehensive price pattern
            price_matches = re.findall(r'\\$([0-9,]+\\.?[0-9]*)', all_text)
            for match in price_matches[:30]:  # Limit to first 30 matches
                try:
                    price = float(match.replace(',', ''))
                    if 5 <= price <= 10000:  # Slightly wider range for fallback
                        prices.append(price)
                except ValueError:
                    continue
        
        # Enhanced price filtering for USED products
        if prices:
            prices = list(set(prices))  # Remove duplicates
            prices.sort()
            
            # Filter out unrealistic prices for used items
            # Remove prices that are likely new/retail prices
            filtered_prices = []
            
            if len(prices) >= 3:
                # Calculate basic statistics
                mean_price = np.mean(prices)
                median_price = np.median(prices)
                std_price = np.std(prices)
                
                # For used items, we expect lower prices than retail
                # Remove top 20% of prices as they might be new/overpriced items
                price_95th = np.percentile(prices, 95)
                price_75th = np.percentile(prices, 75)
                
                # Keep prices below 95th percentile to exclude new item prices
                for price in prices:
                    # Exclude suspiciously high prices that might be new items
                    if price <= price_95th:
                        # Also exclude prices that are statistical outliers (too high)
                        if std_price > 0 and abs(price - median_price) <= 2.5 * std_price:
                            filtered_prices.append(price)
                        elif std_price == 0:  # All prices are the same
                            filtered_prices.append(price)
                
                # If we filtered out too many prices, keep more conservative filter
                if len(filtered_prices) < len(prices) * 0.3:  # Keep at least 30% of original prices
                    filtered_prices = [p for p in prices if p <= price_75th]
                
                # Final fallback - if still too few, use original but remove top 10%
                if len(filtered_prices) < 3:
                    price_90th = np.percentile(prices, 90)
                    filtered_prices = [p for p in prices if p <= price_90th]
                
                prices = filtered_prices if filtered_prices else prices
                
                logger.info(f"Price filtering: {len(prices)} used-market prices retained from original data")
            
            # Remove remaining statistical outliers for used market
            if len(prices) >= 5:
                mean_price = np.mean(prices)
                std_price = np.std(prices)
                if std_price > 0:
                    # More conservative outlier removal for used items
                    prices = [p for p in prices if abs(p - mean_price) <= 2 * std_price]
        
        logger.info(f"Found {len(prices)} prices for {item_name}")
        
        return prices[:25]  # Return max 25 prices
        
    except Exception as e:
        logger.error(f"Error scraping eBay prices: {e}")
        return []

def get_multiple_price_sources(item_name, description, condition, detected_condition=None):
    """Get prices from multiple sources for better accuracy"""
    all_prices = []
    
    # Get eBay prices
    ebay_prices = scrape_ebay_prices(item_name, description, condition, detected_condition)
    all_prices.extend(ebay_prices)
    
    # Additional sources can be added here
    # facebook_prices = scrape_facebook_marketplace(item_name, condition, detected_condition)
    # amazon_prices = scrape_amazon_used_prices(item_name, description, condition)
    # craigslist_prices = scrape_craigslist_prices(item_name, condition)
    
    return all_prices

# ===== PRICING AND VALUATION FUNCTIONS =====

def get_regional_price_multiplier(user_location):
    """Get regional price adjustment multiplier"""
    try:
        if not user_location:
            return 1.0  # No adjustment
        
        # Regional multipliers based on cost of living and market demand
        regional_multipliers = {
            # US Major Cities
            'new_york': 1.25, 'san_francisco': 1.30, 'los_angeles': 1.15,
            'chicago': 1.10, 'boston': 1.20, 'seattle': 1.15, 'miami': 1.05,
            'atlanta': 1.00, 'dallas': 0.95, 'houston': 0.95, 'phoenix': 0.90,
            'philadelphia': 1.05, 'washington_dc': 1.15,
            
            # US States (average)
            'california': 1.15, 'new_york_state': 1.15, 'massachusetts': 1.15,
            'washington_state': 1.10, 'connecticut': 1.10, 'new_jersey': 1.10,
            'maryland': 1.05, 'virginia': 1.00, 'texas': 0.95, 'florida': 1.00,
            'arizona': 0.90, 'nevada': 0.95, 'oregon': 1.05,
            
            # International (examples)
            'london': 1.20, 'tokyo': 1.15, 'sydney': 1.10, 'toronto': 1.05,
            'vancouver': 1.10, 'paris': 1.15, 'berlin': 1.00, 'amsterdam': 1.10
        }
        
        location_key = user_location.lower().replace(' ', '_').replace(',', '')
        return regional_multipliers.get(location_key, 1.0)
        
    except Exception as e:
        logger.error(f"Error getting regional multiplier: {e}")
        return 1.0

def get_enhanced_fallback_estimate(item_name, condition, confidence):
    """Enhanced fallback estimates with more categories and better pricing"""
    try:
        item_name_lower = item_name.lower()
        
        # Expanded market estimates with more realistic pricing
        enhanced_estimates = {
            # Electronics
            'laptop': {'Excellent': 350, 'Good': 250, 'Fair': 180, 'Poor': 100},
            'macbook': {'Excellent': 600, 'Good': 450, 'Fair': 320, 'Poor': 180},
            'iphone': {'Excellent': 300, 'Good': 220, 'Fair': 150, 'Poor': 80},
            'ipad': {'Excellent': 250, 'Good': 180, 'Fair': 130, 'Poor': 70},
            'samsung': {'Excellent': 200, 'Good': 140, 'Fair': 100, 'Poor': 60},
            'phone': {'Excellent': 150, 'Good': 100, 'Fair': 70, 'Poor': 35},
            'tablet': {'Excellent': 120, 'Good': 80, 'Fair': 55, 'Poor': 30},
            'monitor': {'Excellent': 150, 'Good': 110, 'Fair': 75, 'Poor': 40},
            'camera': {'Excellent': 220, 'Good': 160, 'Fair': 110, 'Poor': 60},
            'gaming': {'Excellent': 280, 'Good': 200, 'Fair': 140, 'Poor': 80},
            'xbox': {'Excellent': 250, 'Good': 180, 'Fair': 130, 'Poor': 70},
            'playstation': {'Excellent': 300, 'Good': 220, 'Fair': 160, 'Poor': 90},
            'nintendo': {'Excellent': 200, 'Good': 150, 'Fair': 110, 'Poor': 60},
            'headphones': {'Excellent': 80, 'Good': 55, 'Fair': 35, 'Poor': 20},
            'speaker': {'Excellent': 100, 'Good': 70, 'Fair': 45, 'Poor': 25},
            
            # Automotive
            'car': {'Excellent': 12000, 'Good': 8500, 'Fair': 6000, 'Poor': 3500},
            'motorcycle': {'Excellent': 4000, 'Good': 2800, 'Fair': 2000, 'Poor': 1200},
            'bicycle': {'Excellent': 300, 'Good': 200, 'Fair': 130, 'Poor': 70},
            'truck': {'Excellent': 15000, 'Good': 11000, 'Fair': 8000, 'Poor': 5000},
            
            # Furniture & Home
            'sofa': {'Excellent': 400, 'Good': 280, 'Fair': 180, 'Poor': 90},
            'chair': {'Excellent': 120, 'Good': 80, 'Fair': 50, 'Poor': 25},
            'table': {'Excellent': 200, 'Good': 140, 'Fair': 90, 'Poor': 45},
            'bed': {'Excellent': 300, 'Good': 200, 'Fair': 130, 'Poor': 65},
            'dresser': {'Excellent': 250, 'Good': 170, 'Fair': 110, 'Poor': 55},
            'bookshelf': {'Excellent': 100, 'Good': 70, 'Fair': 45, 'Poor': 25},
            'desk': {'Excellent': 150, 'Good': 100, 'Fair': 65, 'Poor': 35},
            
            # Appliances
            'refrigerator': {'Excellent': 600, 'Good': 400, 'Fair': 250, 'Poor': 120},
            'washer': {'Excellent': 400, 'Good': 280, 'Fair': 180, 'Poor': 90},
            'dryer': {'Excellent': 350, 'Good': 240, 'Fair': 160, 'Poor': 80},
            'microwave': {'Excellent': 80, 'Good': 55, 'Fair': 35, 'Poor': 20},
            'dishwasher': {'Excellent': 300, 'Good': 200, 'Fair': 130, 'Poor': 65},
            
            # Fashion & Accessories
            'watch': {'Excellent': 150, 'Good': 100, 'Fair': 65, 'Poor': 30},
            'jewelry': {'Excellent': 200, 'Good': 140, 'Fair': 90, 'Poor': 45},
            'handbag': {'Excellent': 100, 'Good': 70, 'Fair': 45, 'Poor': 25},
            'shoes': {'Excellent': 60, 'Good': 40, 'Fair': 25, 'Poor': 15},
            'clothing': {'Excellent': 30, 'Good': 20, 'Fair': 12, 'Poor': 6},
            
            # Default
            'default': {'Excellent': 100, 'Good': 70, 'Fair': 45, 'Poor': 25}
        }
        
        # Find best matching category
        best_match = 'default'
        for category in enhanced_estimates.keys():
            if category in item_name_lower:
                best_match = category
                break
        
        estimates = enhanced_estimates[best_match]
        base_price = estimates.get(condition, estimates['Fair'])
        
        # Adjust based on confidence
        confidence_multiplier = 0.85 + (confidence * 0.3)  # 0.85 to 1.15 range
        final_estimate = int(base_price * confidence_multiplier)
        
        return final_estimate
        
    except Exception as e:
        logger.error(f"Error in enhanced fallback estimate: {e}")
        return 50

def calculate_data_quality_score(original_prices, cleaned_prices):
    """Calculate quality score for market data"""
    try:
        # Factors affecting data quality
        sample_size_score = min(1.0, len(original_prices) / 20.0)  # Max score at 20+ samples
        
        outlier_ratio = (len(original_prices) - len(cleaned_prices)) / len(original_prices)
        consistency_score = 1.0 - outlier_ratio
        
        # Price spread analysis
        if len(cleaned_prices) > 1:
            cv = np.std(cleaned_prices) / np.mean(cleaned_prices)  # Coefficient of variation
            spread_score = 1.0 / (1.0 + cv)  # Lower CV = higher score
        else:
            spread_score = 0.5
        
        # Weighted overall score
        overall_score = (
            sample_size_score * 0.4 +
            consistency_score * 0.3 +
            spread_score * 0.3
        )
        
        return min(1.0, overall_score)
        
    except Exception as e:
        logger.error(f"Error calculating data quality score: {e}")
        return 0.5

def process_market_data(prices, condition):
    """Advanced processing of market price data"""
    try:
        prices_array = np.array(prices)
        
        # Statistical analysis
        stats = {
            'mean': np.mean(prices_array),
            'median': np.median(prices_array),
            'std': np.std(prices_array),
            'min': np.min(prices_array),
            'max': np.max(prices_array),
            'count': len(prices_array)
        }
        
        # Outlier detection using IQR method
        Q1 = np.percentile(prices_array, 25)
        Q3 = np.percentile(prices_array, 75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        cleaned_prices = prices_array[(prices_array >= lower_bound) & (prices_array <= upper_bound)]
        
        # Recalculate stats with cleaned data
        if len(cleaned_prices) >= 3:
            cleaned_stats = {
                'mean': np.mean(cleaned_prices),
                'median': np.median(cleaned_prices),
                'std': np.std(cleaned_prices),
                'count': len(cleaned_prices)
            }
        else:
            cleaned_stats = stats
        
        # Data quality assessment
        data_quality_score = calculate_data_quality_score(prices_array, cleaned_prices)
        
        return {
            'original_stats': stats,
            'cleaned_stats': cleaned_stats,
            'cleaned_prices': cleaned_prices.tolist(),
            'outliers_removed': len(prices_array) - len(cleaned_prices),
            'data_quality_score': data_quality_score
        }
        
    except Exception as e:
        logger.error(f"Error processing market data: {e}")
        return None

def analyze_market_trends(prices_analysis, condition):
    """Analyze market trends and provide pricing recommendations"""
    try:
        if not prices_analysis:
            return {'recommended_price': 50, 'confidence_range': {'low': 35, 'high': 70}, 'price_volatility': 'High'}
        
        cleaned_stats = prices_analysis['cleaned_stats']
        
        # Condition-based pricing strategy
        condition_multipliers = {
            'Excellent': {'base': 0.85, 'premium': 0.1},  # 85% + up to 10% premium
            'Good': {'base': 0.70, 'premium': 0.05},      # 70% + up to 5% premium
            'Fair': {'base': 0.55, 'premium': 0.0},       # 55% baseline
            'Poor': {'base': 0.35, 'premium': -0.1}       # 35% with potential discount
        }
        
        multiplier = condition_multipliers.get(condition, condition_multipliers['Fair'])
        
        # Base price calculation (use median for stability)
        base_price = cleaned_stats['median']
        
        # Apply condition multiplier
        recommended_price = base_price * multiplier['base']
        
        # Market volatility assessment
        cv = cleaned_stats['std'] / cleaned_stats['mean'] if cleaned_stats['mean'] > 0 else 1.0
        if cv < 0.2:
            volatility = 'Low'
            range_factor = 0.15
        elif cv < 0.4:
            volatility = 'Medium'
            range_factor = 0.25
        else:
            volatility = 'High'
            range_factor = 0.35
        
        # Calculate confidence range
        confidence_range = {
            'low': int(recommended_price * (1 - range_factor)),
            'high': int(recommended_price * (1 + range_factor))
        }
        
        return {
            'recommended_price': int(recommended_price),
            'confidence_range': confidence_range,
            'price_volatility': volatility,
            'market_median': int(cleaned_stats['median']),
            'condition_multiplier': multiplier['base'],
            'coefficient_variation': cv
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market trends: {e}")
        return {'recommended_price': 50, 'confidence_range': {'low': 35, 'high': 70}, 'price_volatility': 'High'}

def calculate_confidence_score(data_points, condition_confidence, volatility, data_quality):
    """Calculate overall confidence score for pricing"""
    try:
        # Data quantity score
        if data_points >= 15:
            quantity_score = 1.0
        elif data_points >= 8:
            quantity_score = 0.8
        elif data_points >= 4:
            quantity_score = 0.6
        else:
            quantity_score = 0.4
        
        # Volatility score
        volatility_scores = {'Low': 1.0, 'Medium': 0.7, 'High': 0.4}
        volatility_score = volatility_scores.get(volatility, 0.5)
        
        # Combined confidence
        overall_confidence = (
            quantity_score * 0.3 +
            condition_confidence * 0.3 +
            volatility_score * 0.2 +
            data_quality * 0.2
        )
        
        # Convert to categorical
        if overall_confidence >= 0.8:
            return 'High'
        elif overall_confidence >= 0.6:
            return 'Medium'
        else:
            return 'Low'
            
    except Exception as e:
        logger.error(f"Error calculating confidence score: {e}")
        return 'Low'

def validate_used_market_price(estimated_price, item_name, condition):
    """Validate if the estimated price is reasonable for used market"""
    try:
        item_name_lower = item_name.lower()
        
        # Typical used market price ranges by category
        used_market_ranges = {
            'laptop': {'min': 50, 'max': 800, 'typical_range': (100, 500)},
            'phone': {'min': 30, 'max': 400, 'typical_range': (50, 250)},
            'tablet': {'min': 40, 'max': 300, 'typical_range': (60, 200)},
            'monitor': {'min': 25, 'max': 250, 'typical_range': (40, 150)},
            'camera': {'min': 50, 'max': 500, 'typical_range': (80, 300)},
            'gaming': {'min': 60, 'max': 600, 'typical_range': (100, 400)},
            'car': {'min': 1000, 'max': 25000, 'typical_range': (3000, 15000)},
            'watch': {'min': 20, 'max': 300, 'typical_range': (40, 150)},
            'headphones': {'min': 15, 'max': 150, 'typical_range': (25, 80)},
            'furniture': {'min': 30, 'max': 500, 'typical_range': (50, 250)},
        }
        
        # Detect item category
        item_category = 'general'
        for category in used_market_ranges.keys():
            if category in item_name_lower:
                item_category = category
                break
        
        validation_result = {
            'is_reasonable': True,
            'category': item_category,
            'warnings': [],
            'suggestions': []
        }
        
        if item_category in used_market_ranges:
            ranges = used_market_ranges[item_category]
            min_price, max_price = ranges['min'], ranges['max']
            typical_min, typical_max = ranges['typical_range']
            
            # Check if price is within absolute bounds
            if estimated_price < min_price:
                validation_result['is_reasonable'] = False
                validation_result['warnings'].append(f"Price ${estimated_price} seems too low for {item_category}. Minimum expected: ${min_price}")
                validation_result['suggestions'].append(f"Consider if item has significant damage or is very old model")
                
            elif estimated_price > max_price:
                validation_result['is_reasonable'] = False
                validation_result['warnings'].append(f"Price ${estimated_price} seems too high for used {item_category}. Maximum typical: ${max_price}")
                validation_result['suggestions'].append(f"Check if this might be a rare/vintage item or if search included new items")
                
            # Check if price is within typical range
            elif estimated_price < typical_min:
                validation_result['warnings'].append(f"Price ${estimated_price} is below typical range (${typical_min}-${typical_max}) for used {item_category}")
                validation_result['suggestions'].append(f"This could indicate poor condition or older model")
                
            elif estimated_price > typical_max:
                validation_result['warnings'].append(f"Price ${estimated_price} is above typical range (${typical_min}-${typical_max}) for used {item_category}")
                validation_result['suggestions'].append(f"This could indicate excellent condition, recent model, or limited data")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating price: {e}")
        return {'is_reasonable': True, 'warnings': [], 'suggestions': []}

def calculate_valuation(prices, condition, issues_text, image_analysis_results, user_location=None, item_name=""):
    """Enhanced valuation calculation with regional adjustments and better market analysis"""
    try:
        # Get AI-detected condition from image analysis
        ai_detected_condition = condition  # Default to user input
        condition_confidence = 0.5
        
        if image_analysis_results:
            # Use the highest confidence detection from all analyzed images
            best_analysis = max(image_analysis_results, key=lambda x: x.get('confidence', 0))
            if best_analysis.get('detected_condition'):
                ai_detected_condition = best_analysis['detected_condition']
                condition_confidence = best_analysis.get('confidence', 0.5)
                logger.info(f"Using AI-detected condition: {ai_detected_condition} (confidence: {condition_confidence:.2f})")
        
        # Advanced market data processing
        if prices:
            prices_analysis = process_market_data(prices, ai_detected_condition)
        else:
            prices_analysis = None
        
        # Regional price adjustments
        regional_multiplier = get_regional_price_multiplier(user_location)
        
        if not prices or len(prices) < 3:
            # Enhanced fallback pricing with more categories and regional adjustment
            base_estimate = get_enhanced_fallback_estimate(
                item_name,
                ai_detected_condition,
                condition_confidence
            )
            
            # Apply regional adjustment
            adjusted_estimate = int(base_estimate * regional_multiplier)
            
            confidence_level = 'Low'
            if condition_confidence > 0.7:
                confidence_level = 'Medium'
            
            return {
                'estimated_value': adjusted_estimate,
                'range_low': int(adjusted_estimate * 0.7),
                'range_high': int(adjusted_estimate * 1.3),
                'confidence': confidence_level,
                'ai_detected_condition': ai_detected_condition,
                'condition_confidence': condition_confidence,
                'regional_multiplier': regional_multiplier,
                'note': f'AI detected condition: {ai_detected_condition}. Regional adjustment: {regional_multiplier:.2f}x. Estimate based on enhanced market analysis (limited data)'
            }
        
        # Enhanced price analysis with market data
        market_analysis = analyze_market_trends(prices_analysis, ai_detected_condition)
        
        # Calculate final estimate
        base_estimate = market_analysis['recommended_price']
        final_estimate = int(base_estimate * regional_multiplier)
        
        # Enhanced confidence calculation
        confidence = calculate_confidence_score(
            len(prices), 
            condition_confidence, 
            market_analysis['price_volatility'],
            prices_analysis['data_quality_score']
        )
        
        # Price validation with enhanced checks
        validation = validate_used_market_price(
            final_estimate, 
            item_name,
            ai_detected_condition
        )
        
        return {
            'estimated_value': final_estimate,
            'range_low': market_analysis['confidence_range']['low'],
            'range_high': market_analysis['confidence_range']['high'],
            'confidence': confidence,
            'ai_detected_condition': ai_detected_condition,
            'condition_confidence': condition_confidence,
            'market_data_points': len(prices),
            'regional_multiplier': regional_multiplier,
            'market_analysis': market_analysis,
            'price_validation': validation,
            'note': f'Enhanced market analysis with {len(prices)} data points. Regional adjustment: {regional_multiplier:.2f}x. Condition: {ai_detected_condition}'
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced valuation calculation: {e}")
        return {
            'estimated_value': 50,
            'range_low': 30,
            'range_high': 80,
            'confidence': 'Low',
            'note': 'Error occurred during enhanced valuation calculation'
        }

# ===== ENHANCED SEARCH CLASSES =====

@dataclass
class SearchResult:
    price: float
    source: str
    condition: str
    date_sold: Optional[str]
    title: str
    confidence: float

class EnhancedSearchStrategy:
    """Advanced search strategy for better market data collection"""
    
    def __init__(self):
        self.brand_patterns = {
            'apple': r'\\b(apple|iphone|ipad|macbook|imac|mac|airpods)\\b',
            'samsung': r'\\b(samsung|galaxy)\\b',
            'sony': r'\\b(sony|playstation|ps[345])\\b',
            'microsoft': r'\\b(microsoft|xbox|surface)\\b',
            'nintendo': r'\\b(nintendo|switch)\\b',
            'dell': r'\\b(dell|alienware)\\b',
            'hp': r'\\b(hp|hewlett|packard)\\b',
            'lenovo': r'\\b(lenovo|thinkpad)\\b',
            'asus': r'\\b(asus|rog)\\b',
            'acer': r'\\b(acer|predator)\\b'
        }
        
        self.model_patterns = {
            'storage': r'\\b(\\d+(?:gb|tb))\\b',
            'memory': r'\\b(\\d+gb\\s?(?:ram|memory))\\b',
            'screen_size': r'\\b(\\d+(?:\\.\\d+)?"?\\s?inch)\\b',
            'year': r'\\b(20[12]\\d)\\b',
            'generation': r'\\b(gen\\s?\\d+|\\d+(?:st|nd|rd|th)\\s?gen)\\b',
            'processor': r'\\b(i[3579]|ryzen|m1|m2|snapdragon)\\b'
        }
    
    def extract_key_features(self, item_name: str, description: str) -> Dict[str, List[str]]:
        """Extract key features that affect pricing"""
        full_text = f"{item_name} {description}".lower()
        
        features = {
            'brands': [],
            'models': [],
            'specifications': [],
            'keywords': []
        }
        
        # Extract brands
        for brand, pattern in self.brand_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                features['brands'].append(brand)
        
        # Extract specifications
        for spec_type, pattern in self.model_patterns.items():
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                features['specifications'].append(match)
        
        # Extract important keywords
        tech_keywords = re.findall(
            r'\\b(?:unlocked|refurbished|new|mint|sealed|vintage|rare|limited|pro|max|plus|mini|air|ultra)\\b',
            full_text, re.IGNORECASE
        )
        features['keywords'].extend(tech_keywords)
        
        return features

# ===== MAIN ESTIMATION FUNCTION =====

def estimate_item_value(item_name, description, condition, issues="", image_paths=None, user_location=None):
    """
    Main function to estimate item value using all available methods
    
    Args:
        item_name (str): Name of the item
        description (str): Description of the item
        condition (str): Condition ('Excellent', 'Good', 'Fair', 'Poor')
        issues (str): Any issues or defects
        image_paths (list): List of paths to images for analysis
        user_location (str): User's location for regional pricing
    
    Returns:
        dict: Valuation results with estimated value and confidence
    """
    logger.info(f"Starting valuation for: {item_name} ({condition})")
    
    # Initialize image analysis results
    image_analysis_results = []
    
    # Analyze images if provided
    if image_paths:
        for image_path in image_paths:
            try:
                if os.path.exists(image_path):
                    analysis = analyze_image_condition(image_path)
                    image_analysis_results.append(analysis)
                    logger.info(f"Analyzed image: {image_path}")
            except Exception as e:
                logger.error(f"Error analyzing image {image_path}: {e}")
    
    # Get AI-detected condition for enhanced search
    ai_detected_condition = condition  # Default to user input
    if image_analysis_results:
        # Use the highest confidence detection from all analyzed images
        best_analysis = max(image_analysis_results, key=lambda x: x.get('confidence', 0))
        if best_analysis.get('detected_condition'):
            ai_detected_condition = best_analysis['detected_condition']
            logger.info(f"Using AI-detected condition for search: {ai_detected_condition}")
    
    # Scrape market data with AI-detected condition
    logger.info("Scraping market data with AI-enhanced condition detection...")
    scraped_prices = get_multiple_price_sources(item_name, description, condition, ai_detected_condition)
    
    # Calculate valuation
    valuation = calculate_valuation(
        scraped_prices, 
        condition, 
        issues, 
        image_analysis_results, 
        user_location,
        item_name
    )
    
    # Add metadata
    valuation['item_name'] = item_name
    valuation['condition_reported'] = condition
    valuation['ai_detected_condition'] = ai_detected_condition
    valuation['condition_detection_used'] = ai_detected_condition != condition
    valuation['images_analyzed'] = len(image_analysis_results)
    valuation['timestamp'] = datetime.now().isoformat()
    valuation['data_points_found'] = len(scraped_prices)
    valuation['search_success'] = len(scraped_prices) > 0
    
    # Enhanced messaging based on AI detection
    ai_message = ""
    if ai_detected_condition != condition:
        ai_message = f" AI detected condition as '{ai_detected_condition}' (vs reported '{condition}')."
    
    if len(scraped_prices) == 0:
        valuation['message'] = f"No specific market data found for '{item_name}'.{ai_message} Estimate based on AI analysis and general market trends."
    elif len(scraped_prices) < 5:
        valuation['message'] = f"Limited market data found ({len(scraped_prices)} data points).{ai_message} Consider refining item description."
    else:
        valuation['message'] = f"Estimate based on {len(scraped_prices)} market data points.{ai_message}"
    
    logger.info(f"Valuation complete: ${valuation['estimated_value']} ({valuation['confidence']} confidence)")
    
    return valuation