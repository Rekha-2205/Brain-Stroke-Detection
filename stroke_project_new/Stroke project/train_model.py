import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from ml_models.preprocessinimport os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import django

# -------------------- DJANGO SETUP --------------------
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from ml_models.preprocessing import preprocess_ct_image
from ml_models.bilstm_model import BiLSTMStrokeDetector
from admin_panel.models import ModelPerformance, ModelConfiguration, SystemLog


# -------------------- DATA LOADING --------------------
def load_dataset(dataset_path):
    print("Loading CT dataset...")

    stroke_dir = os.path.join(dataset_path, 'stroke')
    non_stroke_dir = os.path.join(dataset_path, 'non_stroke')

    images, labels = [], []

    for label, folder in [(1, stroke_dir), (0, non_stroke_dir)]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")

        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, file)
                img = preprocess_ct_image(img_path, target_size=(224, 224))
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    images = np.expand_dims(images, axis=-1)

    print(f"Total images: {len(images)}")
    print(f"Stroke: {np.sum(labels == 1)}, Normal: {np.sum(labels == 0)}")

    return images, labels


# -------------------- TRAINING --------------------
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, config):
    print("\nBuilding CNN + BiLSTM model...")

    detector = BiLSTMStrokeDetector(
        input_shape=(224, 224, 1),
        bilstm_units=config.bilstm_units,
        dropout_rate=config.dropout_rate
    )

    model = detector.build_model()
    detector.compile_model(learning_rate=config.learning_rate)
    model.summary()

    # -------- CLASS WEIGHTS --------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        verbose=1
    )

    # -------- EVALUATION --------
    probs = model.predict(X_test).ravel()
    preds = (probs >= 0.5).astype(int)

    # Debug: Print predictions and ground truth
    print("\nDEBUG: y_test:", y_test)
    print("DEBUG: preds:", preds)
    print("DEBUG: probs:", probs)
    print(f"DEBUG: y_test unique: {np.unique(y_test, return_counts=True)}")
    print(f"DEBUG: preds unique: {np.unique(preds, return_counts=True)}")

    # Ensure y_test and preds are 1D arrays of 0/1
    y_test_flat = np.asarray(y_test).flatten()
    preds_flat = np.asarray(preds).flatten()

    metrics = {
        'accuracy': accuracy_score(y_test_flat, preds_flat),
        'precision': precision_score(y_test_flat, preds_flat, zero_division=0),
        'recall': recall_score(y_test_flat, preds_flat, zero_division=0),
        'f1_score': f1_score(y_test_flat, preds_flat, zero_division=0),
        'auc_score': roc_auc_score(y_test_flat, probs)
    }

    print("\nTEST PERFORMANCE")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v * 100:.2f}%")

    return detector, history, metrics


# -------------------- PLOTS --------------------
def plot_history(history):
    os.makedirs("media", exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title("Loss")
    plt.legend()

    plt.savefig("media/training_history.png", dpi=300)
    plt.close()


# -------------------- SAVE METRICS --------------------
def save_metrics(metrics, config):
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ModelPerformance.objects.create(
        model_version=version,
        accuracy=metrics['accuracy'] * 100,
        precision=metrics['precision'] * 100,
        recall=metrics['recall'] * 100,
        f1_score=metrics['f1_score'] * 100,
        auc_score=metrics['auc_score'] * 100,
        configuration=config
    )

    return version


# -------------------- MAIN --------------------
def main():
    dataset_path = "media/datasets/ct_scans"
    model_path = "ml_models/trained_models/bilstm_model.h5"

    try:
        config = ModelConfiguration.objects.filter(is_active=True).first()
        if not config:
            raise Exception
    except:
        class Config:
            bilstm_units = 128
            dropout_rate = 0.3
            learning_rate = 0.0001
            batch_size = 16
            epochs = 75
        config = Config()

    X, y = load_dataset(dataset_path)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    detector, history, metrics = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test, config
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    detector.save_model(model_path)

    plot_history(history)
    version = save_metrics(metrics, config)

    SystemLog.objects.create(
        log_type="training",
        user=None,
        description=f"Model {version} trained with accuracy {metrics['accuracy']*100:.2f}%"
    )

    print("\n✅ TRAINING COMPLETE")
    print(f"Model saved at: {model_path}")
    print(f"Version: {version}")


if __name__ == "__main__":
    main()
g import preprocess_ct_image
from ml_models.genetic_algorithm import GeneticAlgorithm
from ml_models.bilstm_model import BiLSTMStrokeDetector
from admin_panel.models import ModelPerformance, ModelConfiguration, SystemLog


def load_dataset(dataset_path):
    """Load CT scan dataset"""
    print("Loading dataset...")
    
    stroke_dir = os.path.join(dataset_path, 'stroke')
    non_stroke_dir = os.path.join(dataset_path, 'non_stroke')
    
    images = []
    labels = []
    
    # Load stroke images
    if os.path.exists(stroke_dir):
        stroke_files = [f for f in os.listdir(stroke_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loading {len(stroke_files)} stroke images...")
        for idx, img_file in enumerate(stroke_files):
            try:
                img_path = os.path.join(stroke_dir, img_file)
                img = preprocess_ct_image(img_path, target_size=(224, 224))
                images.append(img)
                labels.append(1)  # 1 for stroke
                
                if (idx + 1) % 50 == 0:
                    print(f"  Loaded {idx + 1}/{len(stroke_files)} stroke images")
            except Exception as e:
                print(f"  Error loading {img_file}: {e}")
    else:
        print(f"Error: Stroke directory not found: {stroke_dir}")
        return None, None
    
    # Load non-stroke images
    if os.path.exists(non_stroke_dir):
        non_stroke_files = [f for f in os.listdir(non_stroke_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loading {len(non_stroke_files)} non-stroke images...")
        for idx, img_file in enumerate(non_stroke_files):
            try:
                img_path = os.path.join(non_stroke_dir, img_file)
                img = preprocess_ct_image(img_path, target_size=(224, 224))
                images.append(img)
                labels.append(0)  # 0 for non-stroke
                
                if (idx + 1) % 50 == 0:
                    print(f"  Loaded {idx + 1}/{len(non_stroke_files)} non-stroke images")
            except Exception as e:
                print(f"  Error loading {img_file}: {e}")
    else:
        print(f"Error: Non-stroke directory not found: {non_stroke_dir}")
        return None, None
    
    if len(images) == 0:
        print("Error: No images loaded!")
        return None, None
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    
    print(f"\nDataset loaded: {images.shape[0]} images")
    print(f"Stroke cases: {np.sum(labels == 1)}")
    print(f"Non-stroke cases: {np.sum(labels == 0)}")
    
    return images, labels


def train_bilstm_model(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Train BiLSTM model"""
    print("\n" + "="*60)
    print("BILSTM MODEL TRAINING")
    print("="*60)
    
    # Build model
    detector = BiLSTMStrokeDetector(
        input_shape=(224, 224, 1),
        bilstm_units=config.bilstm_units,
        dropout_rate=config.dropout_rate
    )
    
    model = detector.build_model()
    detector.compile_model(learning_rate=config.learning_rate)
    
    print(f"\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining model...")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    
    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=config.batch_size,
        epochs=config.epochs
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = detector.model.predict(X_test, verbose=0)
    test_predictions_binary = (test_predictions > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_predictions_binary)
    precision = precision_score(y_test, test_predictions_binary, zero_division=0)
    recall = recall_score(y_test, test_predictions_binary, zero_division=0)
    f1 = f1_score(y_test, test_predictions_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, test_predictions)
    except:
        auc = 0.0
        print("Warning: Could not calculate AUC score")
    
    print(f"\n{'='*60}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"{'='*60}")
    
    return detector, history, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc
    }


def plot_training_history(history, save_path='media/training_history.png'):
    """Plot training history"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get available metrics
    available_metrics = history.history.keys()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    if 'accuracy' in available_metrics:
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        if 'val_accuracy' in available_metrics:
            axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    if 'loss' in available_metrics:
        axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in available_metrics:
            axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Training history plot saved to: {save_path}")


def save_model_performance(metrics, config):
    """Save model performance to database"""
    try:
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        performance = ModelPerformance.objects.create(
            model_version=model_version,
            accuracy=metrics['accuracy'] * 100,
            precision=metrics['precision'] * 100,
            recall=metrics['recall'] * 100,
            f1_score=metrics['f1_score'] * 100,
            auc_score=metrics['auc_score'] * 100,
            configuration=config
        )
        
        print(f"\n✓ Model performance saved to database")
        print(f"  Model version: {model_version}")
        return model_version
    except Exception as e:
        print(f"\n✗ Error saving performance to database: {e}")
        return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("STROKE DETECTION MODEL TRAINING PIPELINE")
    print("Using Real Hospital CT Scan Dataset")
    print("="*60)
    
    # Configuration
    dataset_path = 'media/datasets/ct_scans'
    model_save_path = 'ml_models/trained_models/bilstm_model.h5'
    
    # Get or create model configuration
    try:
        config = ModelConfiguration.objects.filter(is_active=True).first()
        if not config:
            print("\nCreating default configuration...")
            config = ModelConfiguration.objects.create(
                config_name="Default Configuration",
                population_size=50,
                mutation_rate=0.1,
                crossover_rate=0.8,
                num_generations=100,
                bilstm_units=128,
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=16,  # Reduced for better stability
                epochs=50,
                is_active=True
            )
            print("✓ Created default configuration")
        else:
            print(f"\n✓ Using configuration: {config.config_name}")
    except Exception as e:
        print(f"✗ Error with database: {e}")
        print("Continuing with hardcoded configuration...")
        
        # Create a simple config object
        class SimpleConfig:
            bilstm_units = 128
            dropout_rate = 0.3
            learning_rate = 0.001
            batch_size = 16
            epochs = 50
        
        config = SimpleConfig()
    
    # Step 1: Load Dataset
    print("\n" + "="*60)
    print("STEP 1: LOADING DATASET")
    print("="*60)
    
    images, labels = load_dataset(dataset_path)
    
    if images is None or labels is None:
        print("\n✗ Error: Failed to load dataset!")
        print("\nPlease ensure:")
        print(f"1. Dataset exists at: {dataset_path}")
        print("2. Folder structure:")
        print(f"   {dataset_path}/stroke/     (stroke CT scans)")
        print(f"   {dataset_path}/non_stroke/ (normal CT scans)")
        print("\n3. Generate dataset: python download_real_hospital_ct_dataset.py")
        return
    
    # Step 2: Split Dataset
    print("\n" + "="*60)
    print("STEP 2: SPLITTING DATASET")
    print("="*60)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set:   {X_train.shape[0]} samples ({np.sum(y_train==1)} stroke, {np.sum(y_train==0)} normal)")
    print(f"Validation set: {X_val.shape[0]} samples ({np.sum(y_val==1)} stroke, {np.sum(y_val==0)} normal)")
    print(f"Test set:       {X_test.shape[0]} samples ({np.sum(y_test==1)} stroke, {np.sum(y_test==0)} normal)")
    
    # Step 3: Train BiLSTM Model
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL")
    print("="*60)
    
    detector, history, metrics = train_bilstm_model(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        config
    )
    
    # Step 4: Save Model
    print("\n" + "="*60)
    print("STEP 4: SAVING MODEL")
    print("="*60)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    detector.save_model(model_save_path)
    print(f"✓ Model saved to: {model_save_path}")
    
    # Step 5: Plot Training History
    print("\n" + "="*60)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_training_history(history)
    
    # Step 6: Save Performance Metrics
    print("\n" + "="*60)
    print("STEP 6: SAVING PERFORMANCE METRICS")
    print("="*60)
    
    model_version = save_model_performance(metrics, config)
    
    # Step 7: Log Training
    try:
        SystemLog.objects.create(
            log_type='training',
            user=None,
            description=f'Model training completed. Version: {model_version}, Accuracy: {metrics["accuracy"]*100:.2f}%'
        )
        print("✓ Training logged to system")
    except Exception as e:
        print(f"Warning: Could not log to database: {e}")
    
    # Final Summary
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel Information:")
    print(f"  Version: {model_version}")
    print(f"  Path: {model_save_path}")
    print(f"\nPerformance:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"\nNext Steps:")
    print(f"  1. View training plot: media/training_history.png")
    print(f"  2. Start web server: python manage.py runserver")
    print(f"  3. Upload CT scans for detection")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
