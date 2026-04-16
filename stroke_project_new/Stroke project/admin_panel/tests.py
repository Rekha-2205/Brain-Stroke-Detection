from django.test import TestCase
import numpy as np


class ModelPerformanceTests(TestCase):
    def test_history_field_can_store_json(self):
       
        hist = {'loss': [0.5, 0.4], 'accuracy': [0.6, 0.7]}
        perf = ModelPerformance.objects.create(
            model_version='vtest',
            accuracy=65.0,
            precision=60.0,
            recall=70.0,
            f1_score=65.0,
            auc_score=80.0,
            dataset_name='dummy',
            history=hist
        )
        self.assertIsNotNone(perf.history)
        self.assertEqual(perf.history['loss'], hist['loss'])
        self.assertEqual(perf.history['accuracy'], hist['accuracy'])


class TrainingHandlerStateTests(TestCase):
    def test_state_metrics_reset_and_update(self):
        # reset state and simulate metrics assignment
        training_handler.reset_training_state()
        self.assertIsNone(training_handler.get_training_state().get('latest_metrics'))
        # simulate writing metrics
        training_handler.training_state['latest_metrics'] = {'accuracy': 90.0}
        state = training_handler.get_training_state()
        self.assertEqual(state['latest_metrics']['accuracy'], 90.0)


class DatasetUploadTests(TestCase):
    def setUp(self):
        # create an admin user to authenticate (CustomUser requires role)
        from accounts.models import CustomUser
        self.admin = CustomUser.objects.create(username='admin', role='admin', email='admin@example.com')
        self.admin.set_password('pass')
        self.admin.is_staff = True
        self.admin.is_superuser = True
        self.admin.save()
        self.client.login(username='admin', password='pass')

    def test_upload_with_missing_path_stores_record(self):
        # provide a path that does not exist on disk
        path = 'nonexistent_folder/subdir'
        response = self.client.post(
            '/admin-panel/datasets/upload/',
            {'name': 'testds', 'description': 'desc', 'file_path': path}
        )
        # should redirect after successful save
        self.assertEqual(response.status_code, 302)
        # record should exist in database with same path value
        from .models import Dataset
        ds = Dataset.objects.get(name='testds')
        self.assertEqual(ds.file_path, path)
        # since folder doesn't exist, counts remain zero
        self.assertEqual(ds.total_images, 0)
        self.assertEqual(ds.stroke_images, 0)
        self.assertEqual(ds.non_stroke_images, 0)

    def test_validate_path_endpoint(self):
        # check that AJAX endpoint returns JSON
        resp = self.client.post('/admin-panel/datasets/validate-path/', {'file_path': 'foo/bar'})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('provided', data)
        self.assertIn('resolved', data)
        self.assertIn('exists', data)

    def test_training_aborts_on_one_class(self):
        # verify helper throws when only one class present
        import admin_panel.training_handler as th
        arr = np.zeros(10)
        with self.assertRaises(ValueError) as cm:
            th.validate_label_distribution(arr)
        self.assertIn('both stroke and non-stroke', str(cm.exception))

    def test_loader_accepts_nonstandard_folder_names(self):
        import tempfile, shutil
        from unittest.mock import patch
        # create dataset directory with hyphen for non-stroke
        base_dir = tempfile.mkdtemp()
        try:
            stroke_folder = os.path.join(base_dir, 'Stroke')   # case-insensitive match
            non_folder = os.path.join(base_dir, 'non-stroke')
            os.makedirs(stroke_folder, exist_ok=True)
            os.makedirs(non_folder, exist_ok=True)
            # create dummy files
            open(os.path.join(stroke_folder, 's1.png'), 'a').close()
            open(os.path.join(non_folder, 'n1.png'), 'a').close()
            # patch preprocess to avoid cv2 dependency
            with patch('admin_panel.training_handler.preprocess_ct_image', lambda p: np.zeros((224,224), np.float32)):
                from admin_panel.training_handler import load_dataset_from_path
                X, y = load_dataset_from_path(base_dir)
                self.assertEqual(len(X), 2)
                self.assertEqual(set(y.tolist()), {0.0, 1.0})
        finally:
            shutil.rmtree(base_dir)

    def test_bilstm_train_signature(self):
        # ensure train method accepts class weights and runs a single epoch
        from ml_models.bilstm_model import BiLSTMStrokeDetector
        detector = BiLSTMStrokeDetector((224,224,1))
        model = detector.build_model()
        detector.compile_model()
        import numpy as np
        X = np.zeros((4,224,224,1), dtype=np.float32)
        y = np.array([0,1,0,1], dtype=np.float32)
        history = detector.train(X, y, X, y, batch_size=2, epochs=1, class_weight={0:1,1:1})
        self.assertTrue(hasattr(history, 'history'))

from .models import ModelPerformance
from . import training_handler
import json
