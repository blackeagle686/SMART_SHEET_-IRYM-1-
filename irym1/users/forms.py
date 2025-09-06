from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from .models import User_model


#  1. Login Form
class LoginForm(forms.Form):
    username = forms.CharField(max_length=150, label="Username")
    password = forms.CharField(widget=forms.PasswordInput, label="Password")

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        password = cleaned_data.get("password")

        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("Invalid username or password")
            elif not user.is_active:
                raise forms.ValidationError("This account is inactive.")
        return cleaned_data


class RegisterForm(forms.ModelForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'})
    )
    first_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First Name'})
    )
    last_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last Name'})
    )

    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )

    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm Password'})
    )

    class Meta:
        model = User
        fields = ("first_name", "last_name", "username", "email", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get("password1")
        password2 = cleaned_data.get("password2")

        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords do not match.")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        # Hash the password before saving
        user.set_password(self.cleaned_data["password1"])
        user.is_active = True  # make sure the user can log in
        if commit:
            user.save()
        return user
    
class UserSettingsForm(forms.ModelForm):
    # الحقول من الموديل الأساسي User
    first_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'First Name'})
    )
    last_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'Last Name'})
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'placeholder': 'Email'})
    )

    # الحقول من User_model
    age = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={'placeholder': 'Age'})
    )

    language = forms.ChoiceField(
        choices=[('en', 'English'), ('ar', 'Arabic')],
        required=False,
        widget=forms.Select()
    )

    theme = forms.ChoiceField(
        choices=User_model.THEME_CHOICES,
        required=False,
        widget=forms.Select()
    )

    user_image = forms.ImageField(
        required=False,
    )

    class Meta:
        model = User_model  # نبدأ من User لأنه مرتبط بـ User_model
        fields = ['first_name', 'last_name', 'email']

    def __init__(self, *args, **kwargs):
        self.user_model_instance = kwargs.pop('user_model_instance', None)  # نمررها من الفيو
        super().__init__(*args, **kwargs)

        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'

        # خصائص تخص user_image نضيفها يدويًا
        self.fields['user_image'].widget.attrs['class'] = 'form-control'
        self.fields['user_image'].widget.attrs['accept'] = 'image/*'

    def save(self, commit=True):
        user = super().save(commit)
        profile = self.user_model_instance

        if not profile:
            raise ValueError("User_model instance is required to update profile fields.")

        # تحديث البيانات الإضافية
        profile.age = self.cleaned_data.get('age')
        profile.language = self.cleaned_data.get('language')
        profile.theme = self.cleaned_data.get('theme')

        if self.cleaned_data.get('user_image'):
            profile.set_image(self.cleaned_data.get('user_image'))  # يستخدم الفلترة والتحقق اللي عملناها

        profile.save()
        return user
        

class DeleteAccountPasswordValidation(forms.Form):
    username = forms.CharField(max_length=150, label="Username")
    password = forms.CharField(widget=forms.PasswordInput, label="Password")

    def __init__(self, *args, **kwargs):
        self.request_user = kwargs.pop('request_user', None)
        super().__init__(*args, **kwargs)

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        password = cleaned_data.get("password")

        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("Invalid username or password.")
            elif not user.is_active:
                raise forms.ValidationError("This account is inactive.")

            # Ensure the authenticated user is the one logged in
            if self.request_user and user != self.request_user:
                raise forms.ValidationError("You can only delete your own account.")
        return cleaned_data
