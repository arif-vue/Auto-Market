from django.shortcuts import render
from django.contrib.auth.hashers import make_password
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model
from .models import OTP, UserProfile , CustomUser, RequestService, Review, Contact
from .serializers import (
    CustomUserSerializer,
    CustomUserCreateSerializer,
    UserProfileSerializer,
    OTPSerializer,
    LoginSerializer,
    RequestServiceSerializer,
    ReviewSerializer,
    ContactSerializer
)
from rest_framework_simplejwt.tokens import RefreshToken
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
import random

def error_response(code, message="Error", details=None):
    return Response({
        "success": False,
        "message": message,
        "data": details or {}
    }, status=code)

def success_response(message="Success", data=None, code=200):
    return Response({
        "success": True,
        "message": message,
        "data": data or {}
    }, status=code)

def generate_otp():
    return str(random.randint(100000, 999999))

User = get_user_model()

def send_otp_email(email, otp):
    """
    Smart OTP email sending with guaranteed console fallback
    """
    from django.conf import settings
    
    # Console output for testing and debugging
    print(f"📧 Sending OTP to: {email}")
    print(f"📱 OTP: {otp}")
    
    # Check if this is a test email
    test_domains = ['example.com', 'test.com', 'testing.com']
    domain = email.split('@')[-1].lower()
    is_test_email = domain in test_domains
    
    if is_test_email:
        print("ℹ️ Test email domain - Console output only")
        return
    
    # Try to send real email for non-test addresses
    try:
        html_content = render_to_string('otp_email_template.html', {'otp': otp, 'email': email})
        msg = EmailMultiAlternatives(
            subject='Your OTP Code - AutoMarket',
            body=f'Your OTP verification code is: {otp}',
            from_email=settings.EMAIL_HOST_USER or 'noreply@automarket.com',
            to=[email]
        )
        msg.attach_alternative(html_content, "text/html")
        msg.send(fail_silently=False)
        print(f"✅ EMAIL SENT to: {email}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ EMAIL FAILED: {str(e)}")
        print("💡 Use the OTP from console output above")
        print("="*60 + "\n")
        print("="*60)
        print("Email failed - showing OTP in console")
        print("="*60 + "\n")

@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    """
    Register a new user with full_name, email, password, confirm_password
    Sends 6-digit OTP to user email for verification
    """
    serializer = CustomUserCreateSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        # Send OTP for verification
        otp = generate_otp()
        otp_data = {'email': user.email, 'otp': otp}
        otp_serializer = OTPSerializer(data=otp_data)
        if otp_serializer.is_valid():
            otp_serializer.save()
            try:
                send_otp_email(email=user.email, otp=otp)
            except Exception as e:
                return error_response(
                    code=500,
                    message="Failed to send OTP email",
                    details={"error": [str(e)]}
                )
        return success_response(
            message="User registered successfully. Please verify your email with the OTP sent",
            data={
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role
                }
            },
            code=201
        )
    return error_response(code=400, details=serializer.errors)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """
    Login with email and password (only after email verification)
    """
    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data
        refresh = RefreshToken.for_user(user)
        try:
            profile = user.user_profile
        except UserProfile.DoesNotExist:
            profile = UserProfile.objects.create(user=user)
        
        profile_serializer = UserProfileSerializer(profile, context={'request': request})
        return success_response(
            message="Login successful",
            data={
                "access_token": str(refresh.access_token),
                "refresh_token": str(refresh),
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role,
                    "is_verified": user.is_verified
                },
                "profile": profile_serializer.data
            }
        )
    return error_response(code=401, details=serializer.errors)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def list_users(request):
    users = User.objects.all()
    serializer = CustomUserSerializer(users, many=True)
    return success_response(
        message="Users fetched successfully",
        data={"users": serializer.data}
    )

@api_view(['GET', 'PUT'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """
    GET: Shows user profile with full_name and email (read-only)
    PUT: Update profile (only phone, address, profile_picture - NOT name or email)
    """
    try:
        profile = request.user.user_profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    if request.method == 'GET':
        user = request.user
        profile_serializer = UserProfileSerializer(profile, context={'request': request})
        return success_response(
            message="Profile fetched successfully",
            data={
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role,
                    "is_verified": user.is_verified
                },
                "profile": profile_serializer.data
            }
        )

    if request.method == 'PUT':
        # Only allow updating specific fields, not full_name or email
        allowed_fields = ['phone_number', 'address', 'profile_picture']
        update_data = {k: v for k, v in request.data.items() if k in allowed_fields}
        
        # Handle file uploads separately
        if 'profile_picture' in request.FILES:
            update_data['profile_picture'] = request.FILES['profile_picture']
        
        print(f"Update data: {update_data}")  # Debug print
        
        serializer = UserProfileSerializer(profile, data=update_data, partial=True, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return success_response(
                message="Profile updated successfully",
                data={"profile": serializer.data}
            )
        else:
            print(f"Serializer errors: {serializer.errors}")  # Debug print
            return error_response(code=400, message="Validation failed", details=serializer.errors)

@api_view(['POST'])
@permission_classes([AllowAny])
def create_otp(request):
    email = request.data.get('email')
    if not email:
        return error_response(
            code=400,
            details={"email": ["This field is required"]}
        )
    
    try:
        user = User.objects.get(email=email)
        if user.is_verified:
            return error_response(
                code=400,
                details={"email": ["This account is already verified"]}
            )
    except User.DoesNotExist:
        return error_response(
            code=404,
            details={"email": ["No user exists with this email"]}
        )
    
    otp = generate_otp()
    print(f"📧 Creating OTP for: {email}")
    
    # Delete any existing OTP first
    OTP.objects.filter(email=email).delete()
    
    otp_data = {'email': email, 'otp': otp}
    serializer = OTPSerializer(data=otp_data)
    if serializer.is_valid():
        serializer.save()
        print(f"✅ OTP created and saved successfully")
        
        try:
            send_otp_email(email=email, otp=otp)
        except Exception as e:
            print(f"❌ EMAIL FAILED: {str(e)}")
            return error_response(
                code=500,
                message="Failed to send OTP email",
                details={"error": [str(e)]}
            )
        return success_response(
            message="OTP sent to your email",
            code=201
        )
    return error_response(code=400, details=serializer.errors)

@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp_reset(request):
    email = request.data.get('email')
    otp_value = request.data.get('otp')
    
    if not email or not otp_value:
        details = {}
        if not email:
            details["email"] = ["This field is required"]
        if not otp_value:
            details["otp"] = ["This field is required"]
        return error_response(code=400, details=details)
    
    try:
        otp_obj = OTP.objects.get(email=email)
        if otp_obj.otp != otp_value:
            return error_response(
                code=400,
                details={"otp": ["The provided OTP is invalid"]}
            )
        if otp_obj.is_expired():
            return error_response(
                code=400,
                details={"otp": ["The OTP has expired"]}
            )
        return success_response(message="OTP verified successfully")
    except OTP.DoesNotExist:
        return error_response(
            code=404,
            details={"email": ["No OTP found for this email"]})

@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp(request):
    email = request.data.get('email')
    otp_value = request.data.get('otp')
    
    if not email or not otp_value:
        details = {}
        if not email:
            details["email"] = ["This field is required"]
        if not otp_value:
            details["otp"] = ["This field is required"]
        return error_response(code=400, details=details)
    
    # Convert OTP to string and strip whitespace
    otp_value = str(otp_value).strip()
    
    try:
        otp_obj = OTP.objects.get(email=email)
        db_otp = str(otp_obj.otp).strip()
        
        if otp_obj.is_expired():
            otp_obj.delete()  # Clean up expired OTP
            return error_response(
                code=400,
                details={"otp": ["The OTP has expired. Please request a new one"]}
            )
        
        if db_otp != otp_value:
            return error_response(
                code=400,
                details={"otp": ["The provided OTP is invalid"]}
            )
        
        # Verify the user
        try:
            user = User.objects.get(email=email)
            if user.is_verified:
                otp_obj.delete()  # Clean up OTP
                return error_response(
                    code=400,
                    details={"email": ["This account is already verified"]}
                )
            user.is_verified = True
            user.save()
            otp_obj.delete()
            print(f"✅ Email verified successfully for: {email}")
            return success_response(message="Email verified successfully. You can now log in")
        except User.DoesNotExist:
            return error_response(
                code=404,
                details={"email": ["No user exists with this email"]}
            )
    except OTP.DoesNotExist:
        print("❌ NO OTP FOUND")
        return error_response(
            code=404,
            details={"email": ["No OTP found for this email. Please request a new OTP"]}
        )

@api_view(['POST'])
@permission_classes([AllowAny])
def request_password_reset(request):
    email = request.data.get('email')
    if not email:
        return error_response(
            code=400,
            details={"email": ["This field is required"]}
        )
    
    try:
        user = User.objects.get(email=email)
        if not user.is_verified:
            return error_response(
                code=400,
                details={"email": ["Please verify your email before resetting your password"]}
            )
    except User.DoesNotExist:
        return error_response(
            code=404,
            details={"email": ["No user exists with this email"]}
        )

    otp = generate_otp()
    otp_data = {'email': email, 'otp': otp}
    OTP.objects.filter(email=email).delete()
    serializer = OTPSerializer(data=otp_data)
    if serializer.is_valid():
        serializer.save()
        try:
            send_otp_email(email=email, otp=otp)
        except Exception as e:
            return error_response(
                code=500,
                message="Failed to send OTP email",
                details={"error": [str(e)]}
            )
        return success_response(
            message="OTP sent to your email",
            code=201
        )
    return error_response(code=400, details=serializer.errors)

@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password(request):
    email = request.data.get('email')
    otp_value = request.data.get('otp')
    new_password = request.data.get('new_password')

    if not all([email, otp_value, new_password]):
        details = {}
        if not email:
            details["email"] = ["This field is required"]
        
        if not new_password:
            details["new_password"] = ["This field is required"]
        return error_response(code=400, details=details)

    try:
        otp_obj = OTP.objects.get(email=email)
        if otp_obj.otp != otp_value:
            return error_response(
                code=400,
                details={"otp": ["The provided OTP is invalid"]}
            )
       
        user = User.objects.get(email=email)
        if not user.is_verified:
            return error_response(
                code=400,
                details={"email": ["Please verify your email before resetting your password"]}
            )
        try:
            validate_password(new_password, user)
        except ValidationError as e:
            return error_response(
                code=400,
                details={"new_password": e.messages}
            )

        user.set_password(new_password)
        user.save()
        otp_obj.delete()
        return success_response(message='Password reset successful')
    except OTP.DoesNotExist:
        return error_response(
            code=404,
            details={"email": ["No OTP found for this email"]}
        )
    except User.DoesNotExist:
        return error_response(
            code=404,
            details={"email": ["No user exists with this email"]}
        )


@api_view(['POST'])
@permission_classes([AllowAny])  # ✅ No auth required to refresh token
def refresh_token(request):
    """
    Endpoint to refresh JWT tokens.
    """
    refresh_token = request.data.get('refresh_token')
    if not refresh_token:
        return error_response(
            code=400,
            message="Refresh token is required"
        )

    try:
        refresh = RefreshToken(refresh_token)
        new_access = str(refresh.access_token)
        new_refresh = str(refresh)  # new refresh token (if needed)

        return success_response(
            message="Token refreshed successfully",
            data={
                "access_token": new_access,
                "refresh_token": new_refresh
            }
        )
    except Exception as e:
        return error_response(
            code=400,
            message="Failed to refresh token",
            details={"error": str(e)}
        )


def send_admin_email(subject, message):
    """Helper function to send email to admin"""
    try:
        from django.core.mail import send_mail
        send_mail(
            subject,
            message,
            'arif.elixir@gmail.com',  # From email
            ['arif.elixir@gmail.com'],  # To email
            fail_silently=False,
        )
        return True
    except Exception as e:
        print(f"Email sending failed: {str(e)}")
        return False


@api_view(['POST'])
@permission_classes([AllowAny])
def request_service(request):
    """API endpoint for service requests"""
    try:
        serializer = RequestServiceSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            
            # Prepare email content
            subject = "New Service Request Submitted"
            message = f"""
A new service request has been submitted:

Full Name: {instance.full_name}
Email: {instance.email}
Phone Number: {instance.phone_number}
City: {instance.city}
State: {instance.state}
Zip Code: {instance.zip_code}
Service Type: {instance.service_type}
Types of Items: {instance.types_of_items}
Estimated Total Value: {instance.estimated_total_value}
Preferred Timeframe: {instance.preferred_timeframe}
Additional Information: {instance.additional_information or 'None'}

Submitted on: {instance.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send email to admin
            email_sent = send_admin_email(subject, message)
            
            return success_response(
                message="Service request submitted successfully",
                data={
                    "request_id": instance.id,
                    "email_sent": email_sent
                }
            )
        else:
            return error_response(
                code=400,
                message="Invalid data provided",
                details=serializer.errors
            )
    except Exception as e:
        return error_response(
            code=500,
            message="Failed to submit service request",
            details={"error": str(e)}
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def submit_review(request):
    """API endpoint for submitting reviews"""
    try:
        serializer = ReviewSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            
            # Prepare email content
            subject = "New Review Submitted"
            message = f"""
A new review has been submitted:

Your Name: {instance.your_name}
Email: {instance.email}
Rating: {instance.rating} out of 5 stars
Review: {instance.your_review}

Submitted on: {instance.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send email to admin
            email_sent = send_admin_email(subject, message)
            
            return success_response(
                message="Review submitted successfully",
                data={
                    "review_id": instance.id,
                    "email_sent": email_sent
                }
            )
        else:
            return error_response(
                code=400,
                message="Invalid data provided",
                details=serializer.errors
            )
    except Exception as e:
        return error_response(
            code=500,
            message="Failed to submit review",
            details={"error": str(e)}
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def submit_contact(request):
    """API endpoint for contact form submissions"""
    try:
        serializer = ContactSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            
            # Prepare email content
            subject = "New Contact Form Submission"
            message = f"""
A new contact form has been submitted:

Your Name: {instance.your_name}
Your Email: {instance.your_email}
Your Message: {instance.your_message}

Submitted on: {instance.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send email to admin
            email_sent = send_admin_email(subject, message)
            
            return success_response(
                message="Contact form submitted successfully",
                data={
                    "contact_id": instance.id,
                    "email_sent": email_sent
                }
            )
        else:
            return error_response(
                code=400,
                message="Invalid data provided",
                details=serializer.errors
            )
    except Exception as e:
        return error_response(
            code=500,
            message="Failed to submit contact form",
            details={"error": str(e)}
        )
