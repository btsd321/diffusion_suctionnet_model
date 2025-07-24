import unittest
import torch
from src.pointnet2 import PointNet2  # Assuming PointNet2 class is defined in pointnet2.py

class TestPointNet2(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_points = 1024
        self.num_features = 3  # For XYZ coordinates
        self.pointnet2 = PointNet2()

    def test_forward_pass(self):
        input_data = torch.randn(self.batch_size, self.num_points, self.num_features)
        output = self.pointnet2(input_data)
        self.assertEqual(output.shape, (self.batch_size, 128))  # Assuming output feature size is 128

    def test_output_type(self):
        input_data = torch.randn(self.batch_size, self.num_points, self.num_features)
        output = self.pointnet2(input_data)
        self.assertIsInstance(output, torch.Tensor)

    def test_invalid_input_shape(self):
        invalid_input = torch.randn(self.batch_size, self.num_points + 1, self.num_features)  # Invalid shape
        with self.assertRaises(RuntimeError):
            self.pointnet2(invalid_input)

if __name__ == '__main__':
    unittest.main()