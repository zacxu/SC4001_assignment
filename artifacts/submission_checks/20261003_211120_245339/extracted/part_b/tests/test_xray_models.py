import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from assignment_common import TrainingConfig, evaluate, make_loader, restore_checkpoint, seed_everything
from part_b.src.models import (
    ConvBNReLU, EXPECTED_PARAMETERS, MODEL_NAMES, ResidualBlock, build_model, model_from_metadata,
)
from part_b.src.training import train_classifier


class XRayModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        seed_everything(42)

    def test_parameter_counts_and_figure_layer_dimensions(self):
        expected_convs = {
            "b1": [(1, 16, 5, 1), (16, 32, 5, 1), (32, 64, 3, 1)],
            "b2": [(1, 8, 3, 1), (8, 8, 3, 1), (8, 16, 3, 1),
                   (16, 16, 3, 1), (16, 32, 3, 1), (32, 32, 3, 1)],
            "b3": [(1, 16, 3, 1), (16, 16, 3, 1), (16, 16, 3, 1),
                   (16, 32, 3, 2), (32, 32, 3, 1), (16, 32, 1, 2),
                   (32, 64, 3, 2), (64, 64, 3, 1), (32, 64, 1, 2)],
        }
        expected_fc = {"b1": [(1024, 64), (64, 32), (32, 1)], "b2": [(32, 1)], "b3": [(64, 1)]}
        for name in MODEL_NAMES:
            with self.subTest(architecture=name):
                model = build_model(name)
                convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
                layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
                self.assertEqual([(m.in_channels, m.out_channels, m.kernel_size[0], m.stride[0])
                                  for m in convs], expected_convs[name])
                self.assertEqual([(m.in_features, m.out_features) for m in layers], expected_fc[name])
                count = sum(p.numel() for p in model.parameters())
                self.assertEqual(count, EXPECTED_PARAMETERS[name])
                # Independent layer formula includes BN gamma/beta, excludes running buffers.
                formula = sum(ci * co * k * k + 2 * co for ci, co, k, _ in expected_convs[name])
                formula += sum(ci * co + co for ci, co in expected_fc[name])
                self.assertEqual(count, formula)
                self.assertTrue(all(m.bias is None for m in convs))
                self.assertTrue(all(m.bias is not None for m in layers))
                for module in model.modules():
                    if isinstance(module, ConvBNReLU):
                        self.assertEqual([type(child) for child in module.children()],
                                         [nn.Conv2d, nn.BatchNorm2d, nn.ReLU])
                        self.assertEqual(module.conv.padding, (module.conv.kernel_size[0] // 2,) * 2)
                dropout_names = [key for key, m in model.named_modules() if isinstance(m, nn.Dropout)]
                self.assertEqual(dropout_names, ["classifier.dropout1", "classifier.dropout2"] if name == "b1" else [])
                self.assertTrue(all(m.p == .5 for m in model.modules() if isinstance(m, nn.Dropout)))

    def test_spatial_shapes_flatten_and_single_image_logits(self):
        expected = {"b1": [64, 32, 32, 16, 16, 8, 4],
                    "b2": [64, 64, 32, 32, 32, 16, 16, 16, 8, 1],
                    "b3": [64, 32, 32, 16, 8, 1]}
        for name in MODEL_NAMES:
            model = build_model(name).eval()
            sizes, flattened = [], []
            handles = [module.register_forward_hook(lambda m, i, o: sizes.append(o.shape[-1]))
                       for module in model.features.children()]
            handles.append(model.flatten.register_forward_hook(lambda m, i, o: flattened.append(o.shape[1])))
            with torch.no_grad():
                output = model(torch.zeros(1, 1, 64, 64))
            for handle in handles:
                handle.remove()
            self.assertEqual(sizes, expected[name])
            self.assertEqual(flattened, [1024 if name == "b1" else 32 if name == "b2" else 64])
            self.assertEqual(output.shape, (1,))
            with self.assertRaises(ValueError):
                model(torch.zeros(1, 64, 64))

    def test_strided_padding_preserves_projection_sampling_centers(self):
        for size in (32, 31, 16, 15):
            block = ResidualBlock(16, 32, stride=2)
            main, projection = block.conv1.conv, block.shortcut.conv
            with torch.no_grad():
                main.weight.zero_()
                projection.weight.zero_()
                main.weight[0, 0, 1, 1] = projection.weight[0, 0, 0, 0] = 1
                inputs = torch.arange(size * size, dtype=torch.float32).reshape(1, 1, size, size).repeat(1, 16, 1, 1)
                expected = inputs[:, :1, ::2, ::2]
                torch.testing.assert_close(main(inputs)[:, :1], expected, rtol=0, atol=0)
                torch.testing.assert_close(projection(inputs)[:, :1], expected, rtol=0, atol=0)
                self.assertEqual(block(inputs).shape[-2:], ((size + 1) // 2,) * 2)

    def test_fixed_window_averaging_matches_required_outputs_and_gradients(self):
        for name in MODEL_NAMES:
            pool = build_model(name).features.average_pool
            self.assertIsInstance(pool, nn.AvgPool2d)
            inputs = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8).requires_grad_()
            result = pool(inputs)
            expected_shape = (1, 1, 4, 4) if name == "b1" else (1, 1, 1, 1)
            self.assertEqual(result.shape, expected_shape)
            expected = (torch.tensor([[4.5, 6.5, 8.5, 10.5], [20.5, 22.5, 24.5, 26.5],
                                      [36.5, 38.5, 40.5, 42.5], [52.5, 54.5, 56.5, 58.5]])
                        if name == "b1" else torch.tensor([[31.5]]))
            torch.testing.assert_close(result[0, 0], expected, rtol=0, atol=0)
            result.sum().backward()
            torch.testing.assert_close(inputs.grad, torch.full_like(inputs, .25 if name == "b1" else 1/64),
                                       rtol=0, atol=0)

    def test_residual_sum_activation_and_gradients_in_both_paths(self):
        for ci, co, stride, size in ((16, 16, 1, 32), (16, 32, 2, 32), (32, 64, 2, 16)):
            block = ResidualBlock(ci, co, stride=stride).eval()
            inputs = torch.randn(2, ci, size, size, requires_grad=True)
            output = block(inputs)
            expected = torch.relu(block.conv2(block.conv1(inputs)) + block.shortcut(inputs))
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            output.square().mean().backward()
            self.assertGreater(inputs.grad.abs().sum().item(), 0)
            self.assertGreater(block.conv1.conv.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.conv2.conv.weight.grad.abs().sum().item(), 0)
            self.assertIsInstance(block.add_relu, nn.ReLU)
            if stride == 2:
                self.assertIsInstance(block.shortcut, ConvBNReLU)
                self.assertGreater(block.shortcut.conv.weight.grad.abs().sum().item(), 0)
            else:
                self.assertIsInstance(block.shortcut, nn.Identity)

    def test_unrestricted_output_and_metadata_state_roundtrip(self):
        inputs = torch.randn(2, 1, 64, 64)
        for name in MODEL_NAMES:
            model = build_model(name).eval()
            with torch.no_grad():
                model.classifier.output.weight.zero_()
                model.classifier.output.bias.fill_(-2)
                output = model(inputs)
            torch.testing.assert_close(output, torch.full((2,), -2.), rtol=0, atol=0)
            self.assertTrue((torch.sigmoid(output) < .5).all())
            restored = model_from_metadata(model.metadata()).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(restored(inputs), output, rtol=0, atol=0)
            self.assertIsInstance(model.last_convolution, nn.Conv2d)
            invalid = {**model.metadata(), "convolution_bias": True}
            with self.assertRaises(ValueError):
                model_from_metadata(invalid)

    def test_actual_cnn_shared_training_and_safe_checkpoint_restoration(self):
        dataset = TensorDataset(torch.randn(5, 1, 64, 64), torch.tensor([0., 1., 0., 1., 1.]))
        with tempfile.TemporaryDirectory() as temporary:
            model = build_model("b1")
            result = train_classifier(model, make_loader(dataset, 3, training=True),
                validation_loader=make_loader(dataset, 3),
                config=TrainingConfig(learning_rate=.001, max_epochs=2, device="cpu"),
                run_directory=Path(temporary) / "run", model_config=model.metadata())
            checkpoint = torch.load(result.checkpoint_path, weights_only=True)
            restored = model_from_metadata(checkpoint["model_config"])
            state = restore_checkpoint(result.checkpoint_path, restored)
            metrics = evaluate(restored, make_loader(dataset, 3), state["task_object"], "cpu")
            self.assertEqual(metrics, result.selected_metrics["validation"])
            self.assertEqual(metrics["count"], 5)
            self.assertTrue(all(not key.startswith("test_") for row in result.history for key in row))
            for first, second in zip(model.state_dict().values(), restored.state_dict().values()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
