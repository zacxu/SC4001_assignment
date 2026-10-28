import copy
import unittest

import torch
from torch import nn

from assignment_common import seed_everything
from part_a.src.a4 import (
    FEATURE_NAMES, compute_attributions, feature_masks, global_importance,
    make_references, manual_ablation_differences, rank_importance, validate_checkpoint_pair,
)
from part_a.src.data import CATEGORICAL_FEATURES, HousingPreprocessor, REFIT_YEARS, TRAIN_YEARS
from part_a.src.model import HousingMLP
from part_a.src.wide_deep import WideDeepHousingModel


class KnownPriceModel(nn.Module):
    """An additive embedded model with analytically known ten-feature effects."""
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(3, 2) for _ in range(4))
        with torch.no_grad():
            for index, embedding in enumerate(self.embeddings):
                embedding.weight[:, 0] = torch.arange(3) * (index+1)
                embedding.weight[:, 1] = torch.arange(3) * (index+1) * 2
        self.target_mean, self.target_std = 450_000., 10.

    def forward(self, categorical, continuous):
        cat = sum(embedding(categorical[:, index]).sum(1) for index, embedding in enumerate(self.embeddings))
        return cat + (continuous * torch.arange(1, 7)).sum(1)


class A4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.pre = HousingPreprocessor(
            {name: {1:1, 2:2} if name=='month' else {'A':1, 'B':2} for name in CATEGORICAL_FEATURES},
            torch.arange(1, 7, dtype=torch.float32), torch.arange(1, 7, dtype=torch.float32),
            450_000., 10., REFIT_YEARS, 4)
        self.training = torch.tensor([[2, 1, 2, 2], [1, 1, 2, 1], [2, 2, 2, 1], [1, 2, 1, 1]])
        self.references = make_references(self.pre, self.training)

    def test_training_modes_ties_means_and_ten_unique_masks(self):
        torch.testing.assert_close(self.references['categorical'], torch.tensor([[1,1,2,1]]))
        torch.testing.assert_close(self.references['continuous'], torch.zeros(1,6))
        self.assertEqual(self.references['details'][2]['reference_raw'], 'B')
        self.assertEqual(self.references['details'][2]['training_mode_count'], 3)
        self.assertEqual(self.references['details'][4]['reference_raw'], 1.)
        masks = feature_masks()
        torch.testing.assert_close(torch.cat(masks,1), torch.arange(10).reshape(1,10))
        self.assertEqual(self.references['feature_names'], list(FEATURE_NAMES))
        for invalid in (self.training[:3], self.training.float(), torch.zeros_like(self.training)):
            with self.assertRaises(ValueError):
                make_references(self.pre, invalid)
        pre = copy.copy(self.pre)
        object.__setattr__(pre,'fit_years',TRAIN_YEARS)
        with self.assertRaises(ValueError):
            make_references(pre,self.training)

    def test_analytic_ten_input_sgd_effects_and_reference_values_are_zero(self):
        model = KnownPriceModel()
        cats = torch.tensor([[2,2,1,0],[1,1,2,1]])
        cont = torch.tensor([[1.,-2.,3.,0.,-1.,2.],[0.,0.,0.,0.,0.,0.]])
        attrs, predictions = compute_attributions(model,cats,cont,self.references,batch_size=2)
        expected_cat = (cats-self.references['categorical']).double()*torch.arange(1,5)*3*10
        expected_cont = cont.double()*torch.arange(1,7)*10
        expected = torch.cat((expected_cat,expected_cont),1)
        torch.testing.assert_close(attrs,expected,rtol=0,atol=0)
        torch.testing.assert_close(attrs[1],torch.zeros(10,dtype=torch.float64),rtol=0,atol=0)
        self.assertEqual(attrs.shape,(2,10))
        self.assertEqual(predictions.dtype,torch.float64)
        torch.testing.assert_close(attrs,manual_ablation_differences(model,cats,cont,self.references),rtol=0,atol=0)

    def test_batching_tail_perturbation_grouping_and_model_mode_preservation(self):
        model = KnownPriceModel()
        model.train()
        cats = self.training.repeat(2,1)[:7]
        cont = torch.arange(42,dtype=torch.float32).reshape(7,6)
        progress=[]
        small,pred1=compute_attributions(model,cats,cont,self.references,batch_size=3,on_batch=lambda done,total:progress.append((done,total)))
        large,pred2=compute_attributions(model,cats,cont,self.references,batch_size=7,perturbations_per_eval=5)
        torch.testing.assert_close(small,large,rtol=0,atol=0)
        torch.testing.assert_close(pred1,pred2,rtol=0,atol=0)
        self.assertEqual(progress,[(3,7),(6,7),(7,7)])
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            compute_attributions(model,cats[:0],cont[:0],self.references)

    def test_real_baseline_and_wide_model_manual_differences_and_price_scale(self):
        cats=self.training
        cont=torch.arange(24,dtype=torch.float32).reshape(4,6)/12
        for constructor in (HousingMLP,WideDeepHousingModel):
            seed_everything(42)
            model=constructor(self.pre.cardinalities,self.pre.target_mean,self.pre.target_std,hidden_width=8,embedding_dimension=2)
            attrs,_=compute_attributions(model,cats,cont,self.references,batch_size=4)
            manual=manual_ablation_differences(model,cats,cont,self.references)
            torch.testing.assert_close(attrs,manual,rtol=0,atol=1e-8)
            original_mean=model.target_mean
            model.target_mean+=1_000_000.
            shifted,_=compute_attributions(model,cats,cont,self.references,batch_size=4)
            torch.testing.assert_close(attrs,shifted,rtol=0,atol=1e-8)
            model.target_mean=original_mean
            model.target_std*=2
            scaled,_=compute_attributions(model,cats,cont,self.references,batch_size=4)
            torch.testing.assert_close(scaled,attrs*2,rtol=0,atol=1e-8)

    def test_mean_absolute_aggregation_signed_cancellation_and_stable_ranking(self):
        attrs=torch.tensor([[100.,-40.,0.,0.,0.,0.,0.,0.,0.,0.],[-100.,20.,0.,0.,0.,0.,0.,0.,0.,0.]])
        scores=global_importance(attrs)
        self.assertEqual(scores[0].item(),100.)
        self.assertEqual(scores[1].item(),30.)
        self.assertEqual(attrs[:,0].mean().abs().item(),0.)
        ranked=rank_importance(scores)
        self.assertEqual([row['feature'] for row in ranked[:3]],list(FEATURE_NAMES[:3]))
        for invalid in (attrs[:0],attrs[:,:9],torch.full_like(attrs,float('nan'))):
            with self.assertRaises(ValueError):
                global_importance(invalid)

    def test_pair_guards_same_seed_epoch_preprocessing_architecture_and_lambda(self):
        frozen={'E_star':4,'hidden_width':8,'embedding_dimension':2}
        a3={**frozen,'lambda':2.}
        baseline=HousingMLP(self.pre.cardinalities,self.pre.target_mean,self.pre.target_std,hidden_width=8,embedding_dimension=2)
        wide=WideDeepHousingModel(self.pre.cardinalities,self.pre.target_mean,self.pre.target_std,hidden_width=8,embedding_dimension=2,lambda_value=2.)
        def payload(model,mode):
            return {'epoch':4,'training_config':{'seed':42,'mode':mode},
                    'task':{'kind':'regression','target_mean':self.pre.target_mean,'target_std':self.pre.target_std},
                    'preprocessing':self.pre.metadata(),'model_config':model.metadata()}
        first,second=payload(baseline,'a1_refit'),payload(wide,'fixed_refit')
        validate_checkpoint_pair(first,second,self.pre,frozen,a3)
        for field,value in (('seed',43),('mode','selection')):
            altered=copy.deepcopy(second)
            altered['training_config'][field]=value
            with self.assertRaises(ValueError):
                validate_checkpoint_pair(first,altered,self.pre,frozen,a3)
        for field,value in (('lambda',1.),('architecture','four_embedding_mlp')):
            altered=copy.deepcopy(second)
            altered['model_config'][field]=value
            with self.assertRaises(ValueError):
                validate_checkpoint_pair(first,altered,self.pre,frozen,a3)
        altered=copy.deepcopy(second)
        altered['epoch']=3
        with self.assertRaises(ValueError):
            validate_checkpoint_pair(first,altered,self.pre,frozen,a3)


if __name__=='__main__':
    unittest.main()
