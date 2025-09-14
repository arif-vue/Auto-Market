from rest_framework import serializers
from .models import CustomUser, OTP, UserProfile, RequestService, Review, Contact
from django.contrib.auth import get_user_model, authenticate

User = get_user_model()

class CustomUserSerializer(serializers.ModelSerializer):
    user_profile = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ['id', 'email', 'full_name', 'role', 'is_verified', 'user_profile']
        read_only_fields = ['id', 'is_active', 'is_staff', 'is_superuser']

    def get_user_profile(self, obj):
        try:
            profile = obj.user_profile
            return UserProfileSerializer(profile, context=self.context).data
        except UserProfile.DoesNotExist:
            return None

class CustomUserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True, required=True)
    full_name = serializers.CharField(write_only=True, required=True, max_length=255)
    email = serializers.EmailField(required=True)

    class Meta:
        model = User
        fields = ['id', 'email', 'full_name', 'password', 'confirm_password', 'role']
        extra_kwargs = {
            'email': {'required': True},
            'password': {'required': True, 'write_only': True},
            'role': {'default': 'user'}
        }

    def validate(self, data):
        errors = {}
        
        # Required field validation
        if not data.get('email'):
            errors['email'] = ['This field is required']
        if not data.get('password'):
            errors['password'] = ['This field is required']
        if not data.get('confirm_password'):
            errors['confirm_password'] = ['This field is required']
        if not data.get('full_name'):
            errors['full_name'] = ['This field is required']
            
        # Password validation
        if data.get('password') and data.get('confirm_password'):
            if data['password'] != data['confirm_password']:
                errors['confirm_password'] = ['Password and confirm password do not match']
            if len(data['password']) < 8:
                errors['password'] = ['Password must be at least 8 characters long']
        
        # Email uniqueness validation
        if data.get('email') and User.objects.filter(email=data['email'], is_verified=True).exists():
            errors['email'] = ['A user with this email already exists']
        
        # Role validation - only allow 'user' role for registration
        if data.get('role') and data.get('role') not in ['user']:
            errors['role'] = ['Only user role is allowed during registration']
        
        if errors:
            raise serializers.ValidationError(errors)
        return data

    def create(self, validated_data):
        # Remove confirm_password as it's not needed for user creation
        validated_data.pop('confirm_password', None)
        
        # Delete any unverified users with the same email
        User.objects.filter(email=validated_data['email'], is_verified=False).delete()
        
        user = User.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            full_name=validated_data['full_name'],
            role=validated_data.get('role', 'user')
        )
        
        # Create user profile
        UserProfile.objects.create(user=user)
        return user

class OTPSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(required=True)
    otp = serializers.CharField(required=True)

    class Meta:
        model = OTP
        fields = ['id', 'email', 'otp', 'created_at', 'attempts']
        read_only_fields = ['id', 'created_at', 'attempts']

    def validate(self, data):
        errors = {}
        if not data.get('email'):
            errors['email'] = ['This field is required']
        if not data.get('otp'):
            errors['otp'] = ['This field is required']
        if errors:
            raise serializers.ValidationError(errors)
        return data

class UserProfileSerializer(serializers.ModelSerializer):
    # Full name and email come from the user model (read-only)
    full_name = serializers.CharField(source='user.full_name', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    profile_picture_url = serializers.SerializerMethodField()
    
    class Meta:
        model = UserProfile
        fields = ['id', 'user', 'full_name', 'email', 'profile_picture', 'profile_picture_url', 'phone_number', 'address', 'joined_date']
        read_only_fields = ['id', 'user', 'full_name', 'email', 'joined_date', 'profile_picture_url']

    def get_profile_picture_url(self, obj):
        """Return full URL for profile picture"""
        if obj.profile_picture:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.profile_picture.url)
            else:
                # Fallback when no request context
                return f"http://10.10.12.15:8000{obj.profile_picture.url}"
        return None

    def to_representation(self, instance):
        """Override to include profile_picture_url in response"""
        representation = super().to_representation(instance)
        # For backward compatibility, also include profile_picture as the URL
        representation['profile_picture'] = representation['profile_picture_url']
        return representation

    def validate(self, data):
        # Since we're only allowing profile picture, phone_number, and address to be updated
        # No specific validation needed for these fields
        return data

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True)

    def validate(self, data):
        errors = {}
        email = data.get('email')
        password = data.get('password')

        if not email:
            errors['email'] = ['This field is required']
        if not password:
            errors['password'] = ['This field is required']
        if errors:
            raise serializers.ValidationError(errors)

        user = authenticate(email=email, password=password)
        if not user:
            errors['credentials'] = ['Invalid email or password']
            raise serializers.ValidationError(errors)
        if not user.is_verified:
            errors['credentials'] = ['Account not verified. Please verify your email with the OTP sent']
            raise serializers.ValidationError(errors)
        return user


class RequestServiceSerializer(serializers.ModelSerializer):
    class Meta:
        model = RequestService
        fields = '__all__'
        read_only_fields = ['created_at']


class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = '__all__'
        read_only_fields = ['created_at']

    def validate_rating(self, value):
        if value < 1 or value > 5:
            raise serializers.ValidationError("Rating must be between 1 and 5.")
        return value


class ContactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contact
        fields = '__all__'
        read_only_fields = ['created_at']