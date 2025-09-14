from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser, UserProfile, OTP, RequestService, Review, Contact


class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ('email', 'full_name', 'role')


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = CustomUser
        fields = ('email', 'full_name', 'role', 'is_active', 'is_staff', 'is_superuser')


class CustomUserAdmin(UserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    model = CustomUser
    list_display = ('email', 'full_name', 'role', 'is_staff', 'is_active', 'is_verified')
    list_filter = ('role', 'is_staff', 'is_active', 'is_superuser', 'is_verified')
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('full_name', 'role', 'is_verified')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login',)}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'full_name', 'password1', 'password2', 'role', 'is_staff', 'is_active', 'is_superuser')}),
    )
    
    search_fields = ('email', 'full_name')
    ordering = ('email',)
    filter_horizontal = ('groups', 'user_permissions',)

    def save_model(self, request, obj, form, change):
        """Ensure password is properly hashed when saving through admin"""
        if not change:  # Creating new user
            if form.cleaned_data.get('password1'):
                obj.set_password(form.cleaned_data['password1'])
        super().save_model(request, obj, form, change)

# Register CustomUser with proper admin
admin.site.register(CustomUser, CustomUserAdmin)

# Register other models
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'get_full_name', 'phone_number')
    search_fields = ('user__email', 'user__full_name', 'phone_number', 'address')
    list_filter = ('joined_date',)
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Contact Information', {
            'fields': ('phone_number', 'address')
        }),
        ('Media', {
            'fields': ('profile_picture',)
        }),
    )
    
    def get_full_name(self, obj):
        return obj.user.full_name if obj.user else "No User"
    get_full_name.short_description = 'Full Name'

@admin.register(OTP)
class OTPAdmin(admin.ModelAdmin):
    list_display = ('email', 'otp', 'created_at', 'attempts')
    list_filter = ('created_at',)
    search_fields = ('email', 'otp')


@admin.register(RequestService)
class RequestServiceAdmin(admin.ModelAdmin):
    list_display = ('full_name', 'email', 'service_type', 'state', 'estimated_total_value', 'created_at')
    list_filter = ('service_type', 'state', 'estimated_total_value', 'preferred_timeframe', 'created_at')
    search_fields = ('full_name', 'email', 'city', 'types_of_items')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Contact Information', {
            'fields': ('full_name', 'email', 'phone_number')
        }),
        ('Location', {
            'fields': ('city', 'state', 'zip_code')
        }),
        ('Service Details', {
            'fields': ('service_type', 'types_of_items', 'estimated_total_value', 'preferred_timeframe')
        }),
        ('Additional Information', {
            'fields': ('additional_information', 'created_at')
        }),
    )


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('your_name', 'email', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('your_name', 'email', 'your_review')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Reviewer Information', {
            'fields': ('your_name', 'email')
        }),
        ('Review Details', {
            'fields': ('rating', 'your_review', 'created_at')
        }),
    )


@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = ('your_name', 'your_email', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('your_name', 'your_email', 'your_message')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Contact Information', {
            'fields': ('your_name', 'your_email')
        }),
        ('Message Details', {
            'fields': ('your_message', 'created_at')
        }),
    )