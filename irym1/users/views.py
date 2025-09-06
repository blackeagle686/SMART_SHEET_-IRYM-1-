from django.shortcuts import render, get_object_or_404, redirect
from .models import User_model
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.conf import settings 
from django.contrib.auth import login, authenticate
from .forms import LoginForm, RegisterForm, UserSettingsForm, DeleteAccountPasswordValidation
from django.contrib import messages
from django.contrib.auth import logout
from django.middleware.csrf import rotate_token
from sheet_models.models import Analysis, Prompt
from core_performance import performance_logger
from django.db.models import Q, Count
# Create your views here.

temps = settings.T_PATH

@login_required
@performance_logger
def user_profile(request):
    user_profile = get_object_or_404(User_model, user=request.user)

    prompts = Prompt.objects.filter(user=request.user)

    # Filter analyses with actual related data (and only last 5)
    analyses = (
        Analysis.objects.filter(prompt__in=prompts)
        .annotate(
            desc_count=Count("descriptive_statistics"),
            corr_count=Count("correlations"),
            quality_count=Count("quality_reports"),
            coldesc_count=Count("column_desciptions"),
        )
        .filter(
            Q(desc_count__gt=0) |
            Q(corr_count__gt=0) |
            Q(quality_count__gt=0) |
            Q(coldesc_count__gt=0)
        )
        .order_by("-created_at")[:5]
    )

    return render(request, temps['user-page'], {
        'user': request.user,
        'profile': user_profile,
        'analyses': analyses,
    }) 
    
@performance_logger
def login_view(request):
    # لو فيه يوزر مسجل دخول، نخرّجه وننشئ CSRF جديد
    if request.user.is_authenticated:
        logout(request)
        rotate_token(request)  # تجديد التوكن بعد تسجيل الخروج

    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            user = authenticate(username=username, password=password)
            if user:
                login(request, user)  # Django يغيّر session ID تلقائياً
                rotate_token(request)  # تجديد CSRF بعد تسجيل الدخول
                return redirect('user_profile')
            else:
                form.add_error(None, "username or password is Wrong !!")
    else:
        form = LoginForm()

    return render(request, temps['login'], {'form': form})

@performance_logger
def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            user_profile = User_model.objects.create(user=user)
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, temps['signup'], {'form': form})

@login_required
@performance_logger
def settings_view(request):
    user = request.user
    # استخدم get_or_create لضمان وجود profile بدون تكرار
    profile, created = User_model.objects.get_or_create(user=user)
    delete_form = DeleteAccountPasswordValidation(request_user=request.user)

    if request.method == 'POST':
        form = UserSettingsForm(request.POST, request.FILES, instance=user, user_model_instance=profile)
        
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully.")
            return redirect('settings')
    else:
        initial = {
            'age': profile.age,
            'language': profile.language,
            'theme': profile.theme,
        }
        form = UserSettingsForm(instance=user, user_model_instance=profile, initial=initial)

    return render(request, temps['settings'], {'form': form, 'delete_form': delete_form})

def logout_view(request):
    logout(request)
    return redirect('login')



@login_required
@performance_logger
def delete_account(request):
    if request.method == 'POST':
        form = DeleteAccountPasswordValidation(request.POST, request_user=request.user)
        if form.is_valid():
            user = request.user
            logout(request)  # end the session
            user.delete()    # now delete the actual account
            return redirect('register')
    else:
        form = DeleteAccountPasswordValidation(request_user=request.user)
    
    return render(request, 'delete_account.html', {'form': form})
