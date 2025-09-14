from django.urls import path
from . import views

urlpatterns = [
    # Authentication endpoints
    path('register/', views.register_user, name='register'),
    path('login/', views.login, name='login'),

    # OTP endpoints for email verification
    path('otp/create/', views.create_otp, name='create_otp'),
    path('otp/verify/', views.verify_otp, name='verify_otp'),
    
    # Password reset endpoints (forgot password)
    path('password-reset/request/', views.request_password_reset, name='request_password_reset'),
    path('reset/otp-verify/', views.verify_otp_reset, name='verify_otp_reset'),
    path('password-reset/confirm/', views.reset_password, name='reset_password'),
    
    # Token management
    path('refresh-token/', views.refresh_token, name='refresh_token'),
    
    # User management
    path('users/', views.list_users, name='list_users'),
    path('profile/', views.user_profile, name='user_profile'),
    
    # Service request, review, and contact endpoints
    path('request-service/', views.request_service, name='request_service'),
    path('submit-review/', views.submit_review, name='submit_review'),
    path('submit-contact/', views.submit_contact, name='submit_contact'),
]
