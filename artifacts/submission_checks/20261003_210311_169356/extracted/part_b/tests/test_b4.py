import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from part_b.src.b1 import state_fingerprint
from part_b.src.b4 import explain_batch, explain_selected, resize_and_normalize, run_b4, sample_positive_rows, validate_frozen
from part_b.src.b4_exports import map_diagnostics
from part_b.src.models import build_model


class AnalyticCAM(nn.Module):
    def __init__(self, output_weights):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1, bias=False)
        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([1., -2.]).reshape(2, 1, 1, 1))
            self.fc.weight.copy_(torch.tensor(output_weights).reshape(1, 2))

    def forward(self, images):
        return self.fc(self.conv(images).mean(dim=(2, 3))).squeeze(1)


class B4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampling_is_repeatable_unique_positive_and_not_prediction_conditioned(self):
        labels = torch.tensor([0., 1., 1., 0., 1., 1., 1., 0., 1.])
        ids = torch.arange(100, 109)
        selected = sample_positive_rows(labels, ids)
        self.assertEqual(selected, sample_positive_rows(labels, ids))
        self.assertEqual(len(set(selected['row_ids'])), 5)
        self.assertEqual(selected['labels'], [1]*5)
        self.assertEqual(selected['eligible_positive_rows'], 6)
        self.assertTrue(all(ids[position] == row for position,row in zip(selected['positions'],selected['row_ids'])))
        for invalid_labels, invalid_ids in ((labels[:4],ids[:4]), (torch.tensor([2.]*9),ids), (labels,torch.zeros(9,dtype=torch.long))):
            with self.assertRaises(ValueError):
                sample_positive_rows(invalid_labels,invalid_ids)

    def test_analytic_positive_logit_cam_and_one_captum_forward(self):
        model=AnalyticCAM([2., -1.]).train()
        images=torch.arange(1.,33.).reshape(2,1,4,4)
        calls=[]
        handle=model.register_forward_hook(lambda module, inputs, output: calls.append(len(output)))
        with torch.no_grad():
            result=explain_batch(model,images,model.conv)
        expected=4*images/16
        torch.testing.assert_close(result['positive_coarse_maps'],expected)
        torch.testing.assert_close(result['signed_coarse_maps'],expected)
        torch.testing.assert_close(result['logits'],4*images.mean(dim=(1,2,3)))
        self.assertEqual(calls,[2]); self.assertTrue(model.training)
        self.assertFalse(images.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        handle.remove()

    def test_negative_evidence_is_removed_and_zero_cam_stays_empty(self):
        model=AnalyticCAM([2., 3.])
        result=explain_batch(model,torch.ones(2,1,4,4),model.conv)
        self.assertTrue((result['signed_coarse_maps']<0).all())
        self.assertEqual(int(torch.count_nonzero(result['positive_coarse_maps'])),0)
        self.assertEqual(int(torch.count_nonzero(result['normalized_maps'])),0)
        self.assertTrue(map_diagnostics(result['upsampled_positive_maps'][0],result['normalized_maps'][0])['empty_positive_map'])

    def test_padding_full_and_tail_batches_preserves_selection_order_and_maps(self):
        model=AnalyticCAM([2.,-1.])
        images=torch.arange(1.,49.).reshape(3,1,4,4)
        direct=explain_batch(model,images,model.conv)
        calls=[]
        handle=model.register_forward_hook(lambda module,inputs,output: calls.append(len(output)))
        grouped=explain_selected(model,images,model.conv,[8,1,9],test_count=10,inference_batch_size=4)
        self.assertEqual(calls,[2,4])
        self.assertEqual(grouped['forward_batch_sizes'],[2,4])
        self.assertEqual(grouped['padding_rows'],3)
        torch.testing.assert_close(grouped['logits'],direct['logits'])
        torch.testing.assert_close(grouped['positive_coarse_maps'],direct['positive_coarse_maps'])
        handle.remove()

    def test_map_scaling_zero_handling_bilinear_size_and_geometric_coordinates(self):
        maps=torch.tensor([[[[0.,1.],[2.,3.]]],[[[0.,10.],[20.,30.]]],[[[0.,0.],[0.,0.]]]])
        full, normalized, peaks=resize_and_normalize(maps)
        self.assertEqual(full.shape,(3,1,64,64))
        torch.testing.assert_close(normalized[0],normalized[1])
        self.assertEqual(peaks.tolist(),[3.,30.,0.])
        self.assertTrue(torch.isfinite(normalized).all())
        one=torch.zeros(1,1,64,64); one[0,0,20,30]=2
        diagnostic=map_diagnostics(one,one/2)
        self.assertEqual((diagnostic['peak_x'],diagnostic['peak_y']),(30,20))
        self.assertEqual(diagnostic['outer_8_pixel_frame_mass_fraction'],0)
        for invalid in (maps-1,maps.clone().fill_(float('nan')),maps.squeeze(1)):
            with self.assertRaises(ValueError):
                resize_and_normalize(invalid)

    def test_three_actual_models_preserve_bn_dropout_state_and_native_grids(self):
        torch.manual_seed(42)
        inputs=torch.randn(2,1,64,64)
        for architecture,side in (('b1',16),('b2',16),('b3',8)):
            model=build_model(architecture).train()
            fingerprint=state_fingerprint(model)
            result=explain_batch(model,inputs,model.last_convolution)
            self.assertEqual(result['positive_coarse_maps'].shape,(2,1,side,side))
            self.assertEqual(result['normalized_maps'].shape,(2,1,64,64))
            self.assertEqual(state_fingerprint(model),fingerprint)
            self.assertTrue(all(module.training for module in model.modules()))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            torch.testing.assert_close(result['positive_coarse_maps'],torch.relu(result['signed_coarse_maps']))

    def test_frozen_architecture_class_assumption_and_data_lineage_enforced(self):
        root=Path(__file__).resolve().parents[1]
        frozen=json.loads((root/'part_b/outputs/b3/frozen_b3_settings.json').read_text())
        hashes={'source_sha256':frozen['source_sha256'],'dataset_sha256':frozen['prepared_dataset_sha256'],
                'preprocessing_sha256':frozen['prepared_preprocessing_sha256']}
        validate_frozen(frozen,'b3',**hashes)
        for key,value in (('architecture','b1'),('class_mapping_status','verified'),('source_sha256','changed'),
                          ('refit',True),('probability_threshold',.6)):
            invalid=copy.deepcopy(frozen); invalid[key]=value
            with self.assertRaises(ValueError):
                validate_frozen(invalid,'b3',**hashes)

    def test_existing_experiment_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_b4(directory,'missing',{})


if __name__=='__main__':
    unittest.main()
