"""
Background model training handler
Trains ML models with progress tracking
"""
import os
import threading
import numpy as np
import cv2
from datetime import datetime
from django.conf import settings
from django.db import connections
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.utils import class_weight
import tensorflow as tf

from .models import ModelPerformance, ModelConfiguration, SystemLog, Dataset
from ml_models.bilstm_model import BiLSTMStrokeDetector
from ml_models.preprocessing import preprocess_ct_image


# Custom Keras callback to check for stop requests
class StopTrainingCallback(tf.keras.callbacks.Callback):
    """Callback to stop training when requested"""
    def on_epoch_end(self, epoch, logs=None):
        if is_stop_requested():
            print("\n[TRAINING] Stop requested - stopping training...")
            self.model.stop_training = True


# Global training state
training_state = {
    'is_training': False,
    'current_message': 'Idle',
    'progress': 0,
    'stop_requested': False,
    # metrics recorded at the end of training (accuracy, precision, etc)
    'latest_metrics': None,
}


def update_training_state(message, progress=None):
    """Update global training state"""
    global training_state
    training_state['current_message'] = message
    if progress is not None:
        training_state['progress'] = progress
    if progress is not None:
        print(f"[TRAINING] {message} - Progress: {progress}%")
    else:
        print(f"[TRAINING] {message}")


def get_training_state():
    """Get current training state"""
    # always return a copy to prevent accidental modification
    return training_state.copy()


def request_stop_training():
    """Request training to stop"""
    global training_state
    training_state['stop_requested'] = True
    update_training_state("Stop requested by user", training_state['progress'])
    return True


def is_stop_requested():
    """Check if training stop was requested"""
    return training_state['stop_requested']


def reset_training_state():
    """Reset training state for new session"""
    global training_state
    training_state['stop_requested'] = False
    training_state['is_training'] = False
    training_state['progress'] = 0
    training_state['current_message'] = 'Idle'


def load_dataset_from_path(dataset_path):
    """Load CT scan images from dataset path

    The dataset directory may use a few different conventions for the "stroke"
    and "non_stroke" subfolders (underscores, hyphens, spaces, case). We
    attempt to locate the expected folders and print diagnostic messages so the
    training log makes it obvious which path was used.
    """
    update_training_state("Loading dataset...", 10)

    # helper to search for acceptable folder names under the given base
    def resolve_subdir(base, candidates):
        # try each candidate literally first
        for name in candidates:
            p = os.path.join(base, name)
            if os.path.isdir(p):
                return p
        # fallback: case-insensitive scan for any dir containing the term
        try:
            for entry in os.listdir(base):
                p = os.path.join(base, entry)
                if os.path.isdir(p):
                    for term in candidates:
                        if term.replace('_','').replace('-','').lower() in entry.replace('_','').replace('-','').lower():
                            return p
        except Exception:
            pass
        return None

    stroke_dir = resolve_subdir(dataset_path, ['stroke'])
    non_stroke_dir = resolve_subdir(dataset_path, ['non_stroke','non-stroke','non stroke','nonstroke'])

    print(f"Resolved stroke_dir: {stroke_dir}")
    print(f"Resolved non_stroke_dir: {non_stroke_dir}")

    images = []
    labels = []

    # Load stroke images (label=1)
    if stroke_dir and os.path.exists(stroke_dir):
        stroke_files = [f for f in os.listdir(stroke_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.bmp'))]
        print(f"Loading {len(stroke_files)} stroke images from {stroke_dir}...")
        for idx, filename in enumerate(stroke_files):
            try:
                img_path = os.path.join(stroke_dir, filename)
                img = preprocess_ct_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(1)
            except Exception as e:
                print(f"Error loading stroke image {filename}: {e}")
                continue
    else:
        print(f"Warning: stroke directory not found at {dataset_path}")

    # Load non-stroke images (label=0)
    if non_stroke_dir and os.path.exists(non_stroke_dir):
        non_stroke_files = [f for f in os.listdir(non_stroke_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.bmp'))]
        print(f"Loading {len(non_stroke_files)} non-stroke images from {non_stroke_dir}...")
        for idx, filename in enumerate(non_stroke_files):
            try:
                img_path = os.path.join(non_stroke_dir, filename)
                img = preprocess_ct_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(0)
            except Exception as e:
                print(f"Error loading non-stroke image {filename}: {e}")
                continue
    else:
        print(f"Warning: non-stroke directory not found at {dataset_path}")
    
    if len(images) == 0:
        raise Exception("No valid images found in dataset")
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # make sure there is a channel dimension for Conv2D (grayscale images)
    if X.ndim == 3:
        # shape currently (N, H, W) -> convert to (N, H, W, 1)
        X = np.expand_dims(X, axis=-1)

    print(f"Dataset loaded: {len(images)} images")
    print(f"Data shape: {X.shape}")
    print(f"Stroke: {np.sum(y)} | Non-stroke: {len(y) - np.sum(y)}")
    
    return X, y



def validate_label_distribution(y_array):
    """Raise if the provided label array does not contain both classes."""
    unique = np.unique(y_array)
    if unique.size < 2:
        raise ValueError("Dataset must contain both stroke and non-stroke examples")


def train_model_background(config_id, dataset_id, user_id):
    """
    Train model in a background thread with real model training
    """
    global training_state
    connections.close_all()
    
    try:
        training_state['is_training'] = True
        update_training_state("Loading dataset...", 10)
        
        # Get configuration and dataset
        config = ModelConfiguration.objects.get(id=config_id)
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Construct full path from relative file_path
        dataset_path = os.path.join(settings.BASE_DIR, dataset.file_path)
        
        print(f"\n{'='*70}")
        print(f"TRAINING SESSION - Real Model Training")
        print(f"{'='*70}")
        print(f"Configuration: {config.config_name}")
        print(f"Dataset: {dataset.name}")
        print(f"Dataset Path: {dataset_path}")
        print(f"Total Images: {dataset.total_images}")
        print(f"Stroke Images: {dataset.stroke_images}")
        print(f"Non-Stroke Images: {dataset.non_stroke_images}")
        print(f"{'='*70}\n")
        
        # Load and preprocess dataset
        X, y = load_dataset_from_path(dataset_path)
        
        # ensure label distribution is valid
        validate_label_distribution(y)
        
        if is_stop_requested():
            raise Exception("Training stopped by user")
        
        update_training_state("Splitting dataset...", 20)
        # perform stratified split to keep class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # compute class weights to help imbalance
        try:
            weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train.flatten()),
                y=y_train.flatten()
            )
            class_weight_dict = {i: w for i, w in enumerate(weights)}
        except Exception:
            class_weight_dict = None
            print("Could not compute class weights, continuing without them")
        # perform simple online augmentation on training data to increase variety
        try:
            from ml_models.augmentation import augment_batch_online
            X_train, y_train = augment_batch_online(X_train, y_train)
            print(f"Augmented training set to {X_train.shape[0]} samples")
        except ImportError:
            pass
        
        update_training_state("Building BiLSTM model...", 30)
        input_shape = X_train.shape[1:]
        model_handler = BiLSTMStrokeDetector(input_shape,
                                             bilstm_units=config.bilstm_units,
                                             dropout_rate=config.dropout_rate)
        model = model_handler.build_model()
        model_handler.compile_model(learning_rate=config.learning_rate)
        
        # use configuration values
        batch_size = config.batch_size or 16
        epochs = config.epochs or 2

        update_training_state("Training model...", 40)
        history = model_handler.train(
            X_train, y_train,
            X_test, y_test,
            batch_size=batch_size,
            epochs=epochs,
            class_weight=class_weight_dict
        )
        
        if is_stop_requested():
            raise Exception("Training stopped by user")
        
        update_training_state("Evaluating model...", 80)
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        accuracy_pct = accuracy_score(y_test_flat, y_pred) * 100
        precision_pct = precision_score(y_test_flat, y_pred) * 100
        recall_pct = recall_score(y_test_flat, y_pred) * 100
        f1_pct = f1_score(y_test_flat, y_pred) * 100
        # calculate AUC only if both classes are present
        try:
            auc_pct = roc_auc_score(y_test_flat, y_pred_prob) * 100
        except ValueError as e:
            # sklearn raises a ValueError when only one class is present
            print(f"AUC computation error: {e}")
            auc_pct = 0.0
            SystemLog.objects.create(
                log_type='training',
                user=None,
                description=f'⚠️ AUC-ROC score could not be computed: {e}'
            )
        
        update_training_state("Saving model...", 90)
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_save_path = os.path.join(settings.BASE_DIR, 'ml_models', 'trained_models', f'bilstm_{model_version}.h5')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        
        update_training_state("Saving to database...", 95)
        # store final metrics in training_state for front-end consumption
        training_state['latest_metrics'] = {
            'accuracy': accuracy_pct,
            'precision': precision_pct,
            'recall': recall_pct,
            'f1_score': f1_pct,
            'auc_score': auc_pct,
            'model_version': model_version,
        }
        # convert Keras history to plain python lists so it can be JSON serialized
        history_data = None
        if history is not None and hasattr(history, 'history'):
            history_data = {}
            for key, vals in history.history.items():
                # ensure all values are simple floats
                try:
                    history_data[key] = [float(x) for x in vals]
                except Exception:
                    history_data[key] = list(vals)

        performance = ModelPerformance.objects.create(
            model_version=model_version,
            accuracy=accuracy_pct,
            precision=precision_pct,
            recall=recall_pct,
            f1_score=f1_pct,
            auc_score=auc_pct,
            configuration=config,
            dataset_name=dataset.name,
            history=history_data
        )
        
        SystemLog.objects.create(
            log_type='training',
            user=None,
            description=f'✅ Model training completed successfully!\n'
                       f'Version: {model_version}\n'
                       f'Accuracy: {accuracy_pct:.2f}%\n'
                       f'Precision: {precision_pct:.2f}%\n'
                       f'Recall: {recall_pct:.2f}%\n'
                       f'F1-Score: {f1_pct:.2f}%\n'
                       f'AUC-ROC: {auc_pct:.2f}%\n'
                       f'Dataset: {dataset.name}\n'
                       f'Configuration: {config.config_name}\n'
                       f'Epochs Trained: {len(history.history["loss"])}'
        )
        
        update_training_state("✅ Model trained successfully!", 100)
        print(f"\n{'='*70}")
        print(f"MODEL TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Accuracy:  {accuracy_pct:.2f}%")
        print(f"Precision: {precision_pct:.2f}%")
        print(f"Recall:    {recall_pct:.2f}%")
        print(f"F1-Score:  {f1_pct:.2f}%")
        print(f"AUC-ROC:   {auc_pct:.2f}%")
        print(f"{'='*70}\n")
        
    except Exception as e:
        error_str = str(e)
        
        if "stopped by user" in error_str.lower():
            print(f"\n{'='*70}")
            print(f"TRAINING STOPPED BY USER")
            print(f"{'='*70}\n")
            
            SystemLog.objects.create(
                log_type='training',
                user=None,
                description=f'⏹️ Model training stopped by user\nProgress: {training_state["progress"]}%'
            )
            
            update_training_state("⏹️ Training stopped by user", training_state['progress'])
        else:
            print(f"\n{'='*70}")
            print(f"ERROR DURING TRAINING")
            print(f"{'='*70}")
            print(f"Error: {error_str}")
            print(f"{'='*70}\n")
            
            import traceback
            traceback.print_exc()
            
            SystemLog.objects.create(
                log_type='error',
                user=None,
                description=f'❌ Model training failed:\n{error_str}\n\nTraceback:\n{traceback.format_exc()}'
            )
            
            update_training_state(f"❌ Training failed: {error_str}", 0)
    
    finally:
        training_state['is_training'] = False
        training_state['stop_requested'] = False
        connections.close_all()


def start_model_training(config_id, dataset_id, user_id):
    """
    Start model training in a background thread
    Returns immediately
    """
    global training_state
    
    if training_state['is_training']:
        return False, "Model training is already in progress"
    
    # Reset state for new training session
    training_state['stop_requested'] = False
    training_state['progress'] = 0
    training_state['current_message'] = 'Idle'
    training_state['latest_metrics'] = None
    
    # Start training in a background thread
    training_thread = threading.Thread(
        target=train_model_background,
        args=(config_id, dataset_id, user_id),
        daemon=True
    )
    training_thread.start()
    
    return True, "Model training started in background"
