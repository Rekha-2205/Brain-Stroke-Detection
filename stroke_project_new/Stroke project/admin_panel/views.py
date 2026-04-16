from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views import View
from django.views.generic import ListView, CreateView, UpdateView, DeleteView
from django.contrib import messages
from django.urls import reverse_lazy
from django.db.models import Count
from django.conf import settings
from django.http import JsonResponse
import os
import json
from .models import Dataset, ModelConfiguration, ModelPerformance, SystemLog
from .forms import DatasetUploadForm, ModelConfigurationForm
from .training_handler import start_model_training, get_training_state, request_stop_training
from accounts.models import CustomUser


class AdminRequiredMixin(UserPassesTestMixin):
    """Mixin to ensure only admins can access"""
    def test_func(self):
        return self.request.user.is_authenticated and self.request.user.role == 'admin'
    
    def handle_no_permission(self):
        messages.error(self.request, 'You do not have permission to access this page.')
        return redirect('detection:dashboard')


class AdminDashboardView(AdminRequiredMixin, View):
    """Admin dashboard"""
    template_name = 'admin_panel/dashboard.html'
    
    def get(self, request):
        total_users = CustomUser.objects.exclude(role='admin').count()
        total_datasets = Dataset.objects.count()
        total_predictions = SystemLog.objects.filter(log_type='prediction').count()
        
        recent_logs = SystemLog.objects.all().order_by('-timestamp')[:10]
        
        latest_performance = ModelPerformance.objects.order_by('-training_date').first()
        
        context = {
            'total_users': total_users,
            'total_datasets': total_datasets,
            'total_predictions': total_predictions,
            'recent_logs': recent_logs,
            'latest_performance': latest_performance,
        }
        return render(request, self.template_name, context)


class DatasetListView(AdminRequiredMixin, ListView):
    """List all datasets"""
    model = Dataset
    template_name = 'admin_panel/dataset_list.html'
    context_object_name = 'datasets'
    paginate_by = 10

class DatasetUploadView(AdminRequiredMixin, View):
    """Upload new dataset"""
    template_name = 'admin_panel/dataset_upload.html'
    
    def get(self, request):
        form = DatasetUploadForm()
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        form = DatasetUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            # Get form data
            dataset_name = form.cleaned_data['name']
            description = form.cleaned_data['description']
            raw_path = form.cleaned_data['file_path']

            # Normalize separators and trim whitespace
            path_str = (raw_path or '').strip()
            path_str = path_str.replace('\\', '/')

            # Determine whether the provided path is absolute or relative
            if os.path.isabs(path_str) or (len(path_str) >= 2 and path_str[1] == ':'):
                # Absolute path provided (Windows drive or POSIX absolute)
                full_path = os.path.normpath(path_str)
                save_path = full_path
            else:
                # Treat as project-relative path (relative to BASE_DIR)
                cleaned = path_str.strip('/ ')
                full_path = os.path.normpath(os.path.join(settings.BASE_DIR, cleaned))
                # store relative path in DB to keep records portable
                save_path = cleaned.replace('\\', '/')

            # At upload time we no longer strictly require the folder to exist.
            # Users might add the dataset later or provide a path that will be created
            # by some other process. We will save whatever path they supply, but log
            # warnings if the directory is missing so they know to check later.
            attempted = []
            candidates = [full_path]
            candidates.append(os.path.normpath(os.path.join(settings.BASE_DIR, path_str)))
            candidates.append(os.path.normpath(os.path.expanduser(path_str)))
            candidates.append(os.path.normpath(path_str.replace('/', os.sep)))

            selected = None
            for p in candidates:
                if p and p not in attempted:
                    attempted.append(p)
                    if os.path.exists(p):
                        selected = p
                        break

            # if we didn't locate anything, try to find a suffix match inside project
            if selected is None:
                suffix = path_str.replace('\\', '/').lstrip('/')
                found = None
                try:
                    for root, dirs, files in os.walk(settings.BASE_DIR):
                        for d in dirs:
                            cand = os.path.normpath(os.path.join(root, d))
                            cand_norm = cand.replace('\\', '/')
                            if cand_norm.endswith(suffix) or cand_norm.endswith('/' + suffix):
                                found = cand
                                break
                        if found:
                            break
                except Exception:
                    found = None

                if found:
                    selected = found
                    attempted.append(found)
                    messages.info(request, f'Found matching dataset folder inside project: {found}')
                else:
                    # still save the path but warn the user that it doesn't exist yet
                    messages.warning(request, '⚠️ Provided dataset path could not be verified on disk.')
                    for t in attempted[:3]:
                        messages.info(request, f'Checked (and not found): {t}')
                    messages.info(request, 'You may create this folder later or correct the path.')
                    # we'll continue without returning

            # if we located something, use it; otherwise keep original
            if selected:
                full_path = selected
            # decide what to save in DB: prefer relative path when inside BASE_DIR
            try:
                base = os.path.normpath(settings.BASE_DIR)
                if os.path.commonpath([base, full_path]) == base:
                    # store project-relative path
                    save_path = os.path.relpath(full_path, base).replace('\\', '/')
                else:
                    # store absolute path
                    save_path = full_path
            except Exception:
                save_path = full_path
            
            # Check for required folder structure
            stroke_path = os.path.join(full_path, 'stroke')
            non_stroke_path = os.path.join(full_path, 'non_stroke')

            stroke_exists = os.path.exists(stroke_path)
            non_stroke_exists = os.path.exists(non_stroke_path)

            # If folders are missing, allow saving the dataset (user may add files later)
            if not stroke_exists or not non_stroke_exists:
                # show a prominent warning but continue to save the dataset record
                if not stroke_exists:
                    messages.warning(request, f'⚠️ "stroke" folder not found at {stroke_path}')
                    messages.info(request, f'Expected location: {save_path}/stroke/')
                if not non_stroke_exists:
                    messages.warning(request, f'⚠️ "non_stroke" folder not found at {non_stroke_path}')
                    messages.info(request, f'Expected location: {save_path}/non_stroke/')
                messages.info(request, 'You can still save this dataset. Add images to the folder later and run training.')
            
            # Count images in each folder
            def count_images(directory):
                """Count valid image files in directory"""
                if not os.path.exists(directory):
                    return 0
                try:
                    files = os.listdir(directory)
                    return len([f for f in files 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.bmp'))])
                except Exception as e:
                    print(f"Error counting images in {directory}: {e}")
                    return 0
            
            stroke_count = count_images(stroke_path)
            non_stroke_count = count_images(non_stroke_path)
            total_images = stroke_count + non_stroke_count
            
            print(f"DEBUG: Stroke path: {stroke_path}")
            print(f"DEBUG: Stroke count: {stroke_count}")
            print(f"DEBUG: Non-stroke path: {non_stroke_path}")
            print(f"DEBUG: Non-stroke count: {non_stroke_count}")
            
            # If no images found, still allow saving (user may add later). Keep counts for reference.
            if total_images == 0:
                messages.info(request, 'ℹ️ No images were found in the provided dataset path at this time.')
                messages.info(request, f'Checked: {stroke_path}')
                messages.info(request, f'Checked: {non_stroke_path}')
                messages.info(request, 'Supported formats: .png, .jpg, .jpeg, .dcm, .bmp')
            else:
                if stroke_count == 0:
                    messages.warning(request, '⚠️ No stroke images found')
                    messages.info(request, f'Add images to: {stroke_path}')
                if non_stroke_count == 0:
                    messages.warning(request, '⚠️ No non-stroke images found')
                    messages.info(request, f'Add images to: {non_stroke_path}')
            
            # Save dataset
            try:
                dataset = form.save(commit=False)
                dataset.uploaded_by = request.user
                dataset.file_path = save_path  # Save normalized/absolute path
                dataset.total_images = total_images
                dataset.stroke_images = stroke_count
                dataset.non_stroke_images = non_stroke_count
                dataset.save()
                
                # Log the upload
                SystemLog.objects.create(
                    log_type='user_action',
                    user=request.user,
                    description=f'Dataset "{dataset_name}" uploaded: {stroke_count} stroke + {non_stroke_count} non-stroke = {total_images} total images'
                )
                
                messages.success(
                    request, 
                    f'✅ Dataset "{dataset_name}" uploaded successfully! (path: {save_path})'
                )
                messages.info(
                    request,
                    f'📊 Found {stroke_count} stroke images and {non_stroke_count} non-stroke images (Total: {total_images})'
                )
                return redirect('admin_panel:dataset_list')
                
            except Exception as e:
                messages.error(request, f'❌ Error saving dataset: {str(e)}')
                print(f"Error saving dataset: {e}")
                import traceback
                traceback.print_exc()
                return render(request, self.template_name, {'form': form})
        
        # Form is invalid
        else:
            messages.error(request, '❌ Please correct the errors below')
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
        
        return render(request, self.template_name, {'form': form})


class ValidateDatasetPathView(AdminRequiredMixin, View):
    """AJAX endpoint to validate a dataset path without saving"""
    def post(self, request):
        path_str = request.POST.get('file_path', '').strip().replace('\\', '/')
        # compute full_path as upload logic does
        if os.path.isabs(path_str) or (len(path_str) >= 2 and path_str[1] == ':'):
            candidate = os.path.normpath(path_str)
        else:
            cleaned = path_str.strip('/ ')
            candidate = os.path.normpath(os.path.join(settings.BASE_DIR, cleaned))
        exists = os.path.exists(candidate)
        return JsonResponse({
            'provided': path_str,
            'resolved': candidate,
            'exists': exists
        })


class ModelConfigurationView(AdminRequiredMixin, CreateView):
    """Create model configuration"""
    model = ModelConfiguration
    form_class = ModelConfigurationForm
    template_name = 'admin_panel/model_configuration.html'
    success_url = reverse_lazy('admin_panel:model_training')
    
    def form_valid(self, form):
        messages.success(self.request, 'Configuration saved successfully!')
        return super().form_valid(form)


class ModelTrainingView(AdminRequiredMixin, View):
    """Model training interface"""
    template_name = 'admin_panel/model_training.html'
    
    def get(self, request):
        configurations = ModelConfiguration.objects.all().order_by('-created_at')
        datasets = Dataset.objects.all()
        training_state = get_training_state()
        
        context = {
            'configurations': configurations,
            'datasets': datasets,
            'is_training': training_state['is_training'],
            'current_message': training_state['current_message'],
            'progress': training_state['progress'],
        }
        return render(request, self.template_name, context)
    
    def post(self, request):
        config_id = request.POST.get('configuration')
        dataset_id = request.POST.get('dataset')
        
        # Validate inputs
        if not config_id or not dataset_id:
            messages.error(request, '❌ Please select both configuration and dataset')
            return redirect('admin_panel:model_training')
        
        try:
            config = ModelConfiguration.objects.get(id=config_id)
            dataset = Dataset.objects.get(id=dataset_id)
        except (ModelConfiguration.DoesNotExist, Dataset.DoesNotExist):
            messages.error(request, '❌ Invalid configuration or dataset selected')
            return redirect('admin_panel:model_training')
        
        # Start training in background
        success, message = start_model_training(config_id, dataset_id, request.user.id)
        
        if success:
            messages.success(request, '✅ Model training started! Check the progress below.')
            SystemLog.objects.create(
                log_type='training',
                user=request.user,
                description=f'Model training initiated\nConfiguration: {config.config_name}\nDataset: {dataset.name}'
            )
        else:
            messages.warning(request, f'⚠️ {message}')
        
        return redirect('admin_panel:model_training')


class TrainingProgressView(AdminRequiredMixin, View):
    """Get training progress as JSON (for AJAX)"""
    
    def get(self, request):
        state = get_training_state()
        data = {
            'is_training': state['is_training'],
            'message': state['current_message'],
            'progress': state['progress'],
        }
        # include final metrics when training has completed
        if state.get('latest_metrics'):
            data['metrics'] = state['latest_metrics']
        return JsonResponse(data)


class StopTrainingView(AdminRequiredMixin, View):
    """Stop current model training"""
    
    def post(self, request):
        request_stop_training()
        return JsonResponse({
            'success': True,
            'message': 'Training stop requested',
        })



class ModelPerformanceView(AdminRequiredMixin, ListView):
    """View model performance metrics"""
    model = ModelPerformance
    template_name = 'admin_panel/model_performance.html'
    context_object_name = 'performances'
    ordering = ['-training_date']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # dump history JSON for each performance record to avoid template parsing issues
        import json
        perfs = context.get('performances')
        if perfs:
            # attach a property to each item (QuerySet results are iterables)
            for perf in perfs:
                try:
                    perf.history_json = json.dumps(perf.history or {})
                except Exception:
                    perf.history_json = '{}'
        # also compute latest_history_json in case template needs access separately
        if perfs and len(perfs) > 0:
            context['latest_history_json'] = perfs[0].history_json
        else:
            context['latest_history_json'] = '{}'
        return context


class UserManagementView(AdminRequiredMixin, ListView):
    """Manage users"""
    model = CustomUser
    template_name = 'admin_panel/user_management.html'
    context_object_name = 'users'
    paginate_by = 20
    
    def get_queryset(self):
        return CustomUser.objects.exclude(role='admin').order_by('-date_joined')


class SystemLogsView(AdminRequiredMixin, ListView):
    """View system logs"""
    model = SystemLog
    template_name = 'admin_panel/system_logs.html'
    context_object_name = 'logs'
    paginate_by = 50
    ordering = ['-timestamp']

class DatasetDeleteView(AdminRequiredMixin, View):
    """Delete a dataset"""
    
    def post(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        dataset_name = dataset.name
        
        try:
            # Log the deletion
            SystemLog.objects.create(
                log_type='user_action',
                user=request.user,
                description=f'Dataset "{dataset_name}" deleted by {request.user.username}'
            )
            
            # Delete the dataset
            dataset.delete()
            
            messages.success(request, f'✅ Dataset "{dataset_name}" deleted successfully!')
        except Exception as e:
            messages.error(request, f'❌ Error deleting dataset: {str(e)}')
        
        return redirect('admin_panel:dataset_list')
    
    def get(self, request, pk):
        # Handle GET request (when clicking delete button)
        return self.post(request, pk)


class DatasetViewDetailView(AdminRequiredMixin, View):
    """View dataset details"""
    
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Build full path
        full_path = os.path.join(settings.BASE_DIR, dataset.file_path)
        stroke_path = os.path.join(full_path, 'stroke')
        non_stroke_path = os.path.join(full_path, 'non_stroke')
        
        # Get sample images
        stroke_samples = []
        non_stroke_samples = []
        
        try:
            if os.path.exists(stroke_path):
                files = [f for f in os.listdir(stroke_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
                stroke_samples = files
            
            if os.path.exists(non_stroke_path):
                files = [f for f in os.listdir(non_stroke_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
                non_stroke_samples = files
        except Exception as e:
            messages.warning(request, f'Could not load sample images: {str(e)}')
        
        context = {
            'dataset': dataset,
            'stroke_samples': stroke_samples,
            'non_stroke_samples': non_stroke_samples,
            'path_exists': os.path.exists(full_path),
        }
        
        return render(request, 'admin_panel/dataset_detail.html', context)