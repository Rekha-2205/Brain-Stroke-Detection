"""
Optimized training script for 95%+ model performance
Uses advanced preprocessing, augmentation, transfer learning, and focal loss
"""
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stroke_detection.settings')
django.setup()

from ml_models.advanced_preprocessing import preprocess_ct_image_advanced, preprocess_batch_advanced
from ml_models.augmentation import augment_batch, get_medical_augmentation_pipeline
from ml_models.enhanced_model import EnhancedBiLSTMStrokeDetector, EnsembleStrokeDetector, FocalLoss


def load_dataset_optimized(dataset_path, advanced_preprocess=True):
    """Load dataset with optimized preprocessing"""
    print("Loading dataset with optimized preprocessing...")
    
    stroke_dir = os.path.join(dataset_path, 'stroke')
    non_stroke_dir = os.path.join(dataset_path, 'non_stroke')
    
    images = []
    labels = []
    
    # Load stroke images
    if os.path.exists(stroke_dir):
        stroke_files = [f for f in os.listdir(stroke_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loading {len(stroke_files)} stroke images...")
        
        stroke_paths = [os.path.join(stroke_dir, f) for f in stroke_files]
        preprocess_fn = preprocess_ct_image_advanced if advanced_preprocess else cv2.imread
        
        for idx, img_path in enumerate(stroke_paths):
            try:
                img = preprocess_ct_image_advanced(img_path) if advanced_preprocess else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(1)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Loaded {idx + 1}/{len(stroke_paths)} stroke images")
            except Exception as e:
                print(f"  Warning: {img_path}: {e}")
    
    # Load non-stroke images
    if os.path.exists(non_stroke_dir):
        non_stroke_files = [f for f in os.listdir(non_stroke_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loading {len(non_stroke_files)} non-stroke images...")
        
        non_stroke_paths = [os.path.join(non_stroke_dir, f) for f in non_stroke_files]
        
        for idx, img_path in enumerate(non_stroke_paths):
            try:
                img = preprocess_ct_image_advanced(img_path) if advanced_preprocess else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(0)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Loaded {idx + 1}/{len(non_stroke_paths)} non-stroke images")
            except Exception as e:
                print(f"  Warning: {img_path}: {e}")
    
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


def train_optimized_model(X_train, y_train, X_val, y_val, X_test, y_test, use_ensemble=False):
    """Train model with all optimizations"""
    print("\n" + "="*70)
    print("OPTIMIZED MODEL TRAINING FOR 95%+ PERFORMANCE")
    print("="*70)
    
    if use_ensemble:
        print("\nUsing Ensemble approach...")
        detector = EnsembleStrokeDetector(
            models_config=[
                {'use_transfer_learning': True, 'bilstm_units': 256},
                {'use_transfer_learning': False, 'bilstm_units': 128},
            ]
        )
        detector.build_ensemble(input_shape=(224, 224, 1))
        
        # Train ensemble
        histories = detector.train_ensemble(
            X_train, y_train, X_val, y_val,
            batch_size=32,
            epochs=100,
            class_weight=None
        )
        
        # Evaluate ensemble
        test_predictions = detector.predict_batch_ensemble(X_test)
    else:
        print("\nUsing single enhanced model with transfer learning...")
        detector = EnhancedBiLSTMStrokeDetector(
            input_shape=(224, 224, 1),
            bilstm_units=256,
            dropout_rate=0.4,
            use_transfer_learning=True
        )
        
        model = detector.build_model()
        detector.compile_model(learning_rate=0.0005, use_focal_loss=True)
        
        print(f"\nModel Architecture:")
        model.summary()
        
        # Calculate class weights for imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        class_weight = {int(u): len(y_train) / (2 * c) for u, c in zip(unique, counts)}
        print(f"\nClass weights: {class_weight}")
        
        # Train model
        print(f"\nTraining with settings:")
        print(f"  - Batch size: 32")
        print(f"  - Epochs: 100")
        print(f"  - Learning rate: 0.0005 (with reduction on plateau)")
        print(f"  - Loss: Focal Loss (for class imbalance)")
        print(f"  - Transfer Learning: ResNet50")
        
        history = detector.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=32,
            epochs=100,
            class_weight=class_weight
        )
        
        # Evaluate on test set
        test_predictions = detector.model.predict(X_test, verbose=0)
        test_predictions = test_predictions.flatten()
    
    # Binary predictions
    test_predictions_binary = (test_predictions > 0.5).astype(int)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, test_predictions_binary)
    precision = precision_score(y_test, test_predictions_binary, zero_division=0)
    recall = recall_score(y_test, test_predictions_binary, zero_division=0)
    f1 = f1_score(y_test, test_predictions_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, test_predictions)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_predictions_binary)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    print(f"\n{'='*70}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy:    {accuracy*100:.2f}%")
    print(f"Precision:   {precision*100:.2f}%")
    print(f"Recall:      {recall*100:.2f}%")
    print(f"Sensitivity: {sensitivity*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")
    print(f"F1-Score:    {f1*100:.2f}%")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"{'='*70}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'timestamp': datetime.now().isoformat()
    }
    
    return detector, metrics, test_predictions


def plot_performance(test_predictions, y_test, save_path='media/enhanced_performance.png'):
    """Plot comprehensive performance metrics"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Enhanced Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_predictions)
    auc = roc_auc_score(y_test, test_predictions)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Prediction distribution
    axes[0, 1].hist(test_predictions[y_test == 0], bins=30, alpha=0.6, label='Non-Stroke')
    axes[0, 1].hist(test_predictions[y_test == 1], bins=30, alpha=0.6, label='Stroke')
    axes[0, 1].set_xlabel('Prediction Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrix
    preds_binary = (test_predictions > 0.5).astype(int)
    cm = confusion_matrix(y_test, preds_binary)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Annotations
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
    
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Non-Stroke', 'Stroke'])
    axes[1, 0].set_yticklabels(['Non-Stroke', 'Stroke'])
    
    # Metrics summary
    accuracy = accuracy_score(y_test, preds_binary)
    precision = precision_score(y_test, preds_binary)
    recall = recall_score(y_test, preds_binary)
    f1 = f1_score(y_test, preds_binary)
    
    metrics_text = f'Accuracy: {accuracy*100:.2f}%\nPrecision: {precision*100:.2f}%\nRecall: {recall*100:.2f}%\nF1-Score: {f1*100:.2f}%'
    axes[1, 1].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Performance Metrics')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPerformance plots saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    dataset_path = 'media/datasets/ct_scans'
    
    # Load dataset
    images, labels = load_dataset_optimized(dataset_path, advanced_preprocess=True)
    
    if images is None:
        print("Failed to load dataset!")
        return
    
    # Data augmentation for training set
    print("\nApplying data augmentation...")
    
    # Split first
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Augment training data
    transform = get_medical_augmentation_pipeline(p=0.8)
    X_train_aug, y_train_aug = augment_batch(X_train, y_train, augmentation_factor=2, transform=transform)
    
    print(f"After augmentation - Train: {X_train_aug.shape[0]}")
    
    # Train model
    detector, metrics, test_predictions = train_optimized_model(
        X_train_aug, y_train_aug,
        X_val, y_val,
        X_test, y_test,
        use_ensemble=False  # Set to True for ensemble
    )
    
    # Save model
    model_path = 'ml_models/trained_models/enhanced_bilstm_model.h5'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    detector.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot results
    plot_performance(test_predictions, y_test)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Target 95%+ Achieved!" if metrics['accuracy'] > 0.95 else "TRAINING COMPLETE - Continue optimization")
    print("="*70)


if __name__ == "__main__":
    main()
