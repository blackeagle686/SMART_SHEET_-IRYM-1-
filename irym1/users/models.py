from django.db import models
from django.contrib.auth.models import User
from django.core.validators import EmailValidator
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
import os
from PIL import Image


class User_model(models.Model):  
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')  # 👈 اسم أوضح

    THEME_CHOICES = [
        ('light', 'Light'),
        ('deep-sea', 'Deep Sea'),
        ('dark', 'Dark'),
        ('burry', 'Burry'),
        ('military', 'Military'),
        ('ice', 'Ice'),
    ]

    user_image = models.ImageField(upload_to='user_images/', default='defult-user-profile-image.png')
    age = models.PositiveIntegerField(null=True, blank=True)
    theme = models.CharField(max_length=20, choices=THEME_CHOICES, default='light')
    language = models.CharField(max_length=10, default='en')
    last_active_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} ({self.user.email})"

    @property
    def display_email(self):
        return self.user.email or f"No email for {self.user.username}"

    @property
    def display_age(self):
        return self.age or f"No age for {self.user.username}"

    @property
    def display_image(self):
        return self.user_image.url if self.user_image else f"No image for {self.user.username}"

    def set_theme(self, theme):
        if theme in dict(self.THEME_CHOICES):
            self.theme = theme
            self.save()
        else:
            raise ValueError("Invalid theme choice")

    def set_age(self, age):
        if isinstance(age, int) and age > 0:
            self.age = age
            self.save()
        else:
            raise ValueError("Age must be a positive integer")

    def set_email(self, email):
        validator = EmailValidator()
        try:
            validator(email)
            self.user.email = email
            self.user.save()
        except ValidationError:
            raise ValueError("Invalid email format")

    def set_image(self, image_file):
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif']

        if not isinstance(image_file, (InMemoryUploadedFile, TemporaryUploadedFile)):
            raise ValidationError("Invalid file type. Please upload a valid image file.")

        ext = os.path.splitext(image_file.name)[1].lower()
        if ext not in allowed_extensions:
            raise ValidationError(f"Unsupported file extension '{ext}'. Allowed: {', '.join(allowed_extensions)}")

        max_size = 5 * 1024 * 1024
        if image_file.size > max_size:
            raise ValidationError("Image file size exceeds the 5MB limit.")

        try:
            img = Image.open(image_file)
            img.verify()
        except Exception:
            raise ValidationError("Corrupted or invalid image file.")

        self.user_image = image_file
        self.save()

    def delete_user(self):
        self.user.delete()
