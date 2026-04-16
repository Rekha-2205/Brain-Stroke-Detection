from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.views import View
from django.views.generic import CreateView, UpdateView
from django.urls import reverse_lazy
from .models import CustomUser
from .forms import UserRegistrationForm, UserLoginForm, UserProfileForm

class UserRegistrationView(View):
    """User registration view"""
    template_name = 'accounts/register.html'
    
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('detection:dashboard')
        form = UserRegistrationForm()
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # Set password using the custom password field
            password = form.cleaned_data['password']
            user.set_password(password)
            user.save()
            messages.success(request, 'Account created successfully! Please login.')
            return redirect('accounts:login')
        return render(request, self.template_name, {'form': form})

class UserLoginView(View):
    """User login view"""
    template_name = 'accounts/login.html'
    
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('detection:dashboard')
        form = UserLoginForm()
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            
            if user is not None:
                if user.is_active:
                    login(request, user)
                    messages.success(request, f'Welcome back, {user.first_name}!')
                    
                    # Redirect based on role
                    if user.role == 'admin':
                        return redirect('admin_panel:dashboard')
                    else:
                        return redirect('detection:dashboard')
                else:
                    messages.error(request, 'Your account has been deactivated.')
            else:
                messages.error(request, 'Invalid username or password.')
        
        return render(request, self.template_name, {'form': form})

class UserLogoutView(LoginRequiredMixin, View):
    """User logout view with confirmation"""
    template_name = 'accounts/logout_confirm.html'
    
    def get(self, request):
        """Display logout confirmation page"""
        return render(request, self.template_name)
    
    def post(self, request):
        """Perform logout after confirmation"""
        logout(request)
        messages.success(request, 'You have been logged out successfully.')
        return redirect('accounts:login')

class UserProfileView(LoginRequiredMixin, UpdateView):
    """User profile management"""
    model = CustomUser
    form_class = UserProfileForm
    template_name = 'accounts/profile.html'
    success_url = reverse_lazy('accounts:profile')
    
    def get_object(self):
        return self.request.user
    
    def form_valid(self, form):
        messages.success(self.request, 'Profile updated successfully!')
        return super().form_valid(form)
