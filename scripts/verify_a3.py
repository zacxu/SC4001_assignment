"""Restore A3 CUDA checkpoints; audit selection/refits without rescoring test data."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from assignment_common import EarlyStopper, evaluate, make_loader, restore_checkpoint, seed_everything
from assignment_common.artifacts import write_json
from part_a.src.a2 import check_same_preprocessing, model_state_fingerprint, sha256_file, validate_frozen_settings
from part_a.src.a3 import LAMBDAS, baseline_state_fingerprint, choose_lambda, summarize_comparison
from part_a.src.data import load_data_bundle
from part_a.src.wide_deep import WideDeepHousingModel


def close_metrics(expected, actual):
    if expected.keys()!=actual.keys():
        raise AssertionError('Metric names differ')
    for key,value in expected.items():
        if value is None:
            assert actual[key] is None
        elif not math.isclose(value,actual[key],rel_tol=1e-9,abs_tol=1e-8):
            raise AssertionError(f'{key} differs: {value} -> {actual[key]}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'part_a/outputs/a3')
    output=parser.parse_args().output.resolve()
    config=json.loads((output/'configuration.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    status=json.loads((output/'status.json').read_text())
    assert summary['status']==status['status']=='completed'
    assert status['search_completed']==status['refits_completed']==5
    a1,a2,prepared=[Path(config[key]) for key in ('a1_directory','a2_directory','prepared_directory')]
    frozen=json.loads((a1/'frozen_a1_settings.json').read_text())
    validate_frozen_settings(frozen,json.loads((a1/'summary.json').read_text()))
    assert frozen==config['frozen_a1_settings']==summary['frozen_a1_settings']
    paths={'a1_frozen':a1/'frozen_a1_settings.json','a1_refit_preprocessing':a1/frozen['refit_preprocessing'],
           'a2_summary':a2/'summary.json','selection_data':prepared/'selection/datasets.pt','refit_data':prepared/'refit/datasets.pt'}
    assert all(sha256_file(path)==config['input_hashes'][key] for key,path in paths.items())
    bundles={name:load_data_bundle(prepared/name,expected_protocol=name) for name in ('selection','refit')}
    for name,bundle in bundles.items():
        check_same_preprocessing(torch.load(output/f'{name}_preprocessing.pt',map_location='cpu',weights_only=True),bundle.preprocessing.metadata())
        assert bundle.source_sha256==config['source_sha256']==frozen['source_sha256']
    check_same_preprocessing(bundles['refit'].preprocessing.metadata(),torch.load(paths['a1_refit_preprocessing'],map_location='cpu',weights_only=True))
    grid=json.loads((output/'lambda_search.json').read_text())
    chosen=choose_lambda(grid)
    assert chosen==summary['selected']==json.loads((output/'selection.json').read_text())['selected']
    a3_frozen=json.loads((output/'frozen_a3_settings.json').read_text())
    assert a3_frozen['lambda']==chosen['lambda'] and a3_frozen['E_star']==frozen['E_star']
    assert all(value is True for value in json.loads((output/'baseline_parity.json').read_text()).values())
    records=json.loads((output/'per_seed_results.json').read_text())
    baseline=json.loads((output/'baseline_results.json').read_text())
    assert baseline==json.loads((a2/'summary.json').read_text())['variants']['baseline']['per_seed']
    groups,paired=summarize_comparison(records,baseline,epoch_budget=frozen['E_star'])
    assert groups==summary['variants'] and paired==summary['paired_differences']
    assert summary['test_assisted_epoch_budget'] is True and summary['test_monitoring_during_a3'] is False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the A3 production audit')
    torch.set_num_threads(4)
    audited=[]
    initial_hashes=set()
    for stage,rows in (('selection',grid),('refit',records)):
        bundle=bundles[stage]
        for row in rows:
            run=output/row['run_directory']
            run_config=json.loads((run/'config.json').read_text())
            run_summary=json.loads((run/'summary.json').read_text())
            history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
            training=run_config['training']
            assert training['learning_rate']==.01 and run_config['batch_size']==512
            assert training['seed']==row['seed'] and training['device']=='cuda'
            assert run_summary['status']=='completed' and run_summary['test_assisted_stopping'] is False
            assert not (run/'internal_test_monitor.jsonl').exists()
            assert not any(key.startswith('test_') for epoch in history for key in epoch)
            assert [epoch['epoch'] for epoch in history]==list(range(1,row['epochs_completed']+1))
            seed_everything(row['seed'])
            model=WideDeepHousingModel.from_metadata(run_config['model_config'])
            assert model_state_fingerprint(model)==row['initial_model_sha256']==run_config['model_config']['initial_model_sha256']
            assert baseline_state_fingerprint(model)==row['baseline_initial_sha256']==run_config['model_config']['baseline_initial_sha256']
            assert model.lambda_value==row['lambda']
            assert len(model.embeddings)==4 and sum(isinstance(m,nn.Embedding) for m in model.modules())==4
            assert model.wide.in_features==model.mlp[0].in_features==22 and model.wide.out_features==1
            assert model.mlp[0].out_features==256 and isinstance(model.mlp[1],nn.LayerNorm) and isinstance(model.mlp[2],nn.ReLU)
            assert sum(p.numel() for p in model.parameters())==row['trainable_parameters']==7088
            assert model.metadata()['active_parameters']==row['active_parameters']
            checkpoint=restore_checkpoint(output/row['checkpoint'],model,device='cuda')
            assert all(group['lr']==.01 and group['weight_decay']==0 for group in checkpoint['optimizer_state']['param_groups'])
            assert checkpoint['training_config']==training and checkpoint['model_config']==run_config['model_config']
            assert checkpoint['epoch']==run_summary['best_epoch']
            check_same_preprocessing(checkpoint['preprocessing'],bundle.preprocessing.metadata())
            actual=evaluate(model,make_loader(bundle.datasets['train'],512,device='cuda'),checkpoint['task_object'],'cuda')
            close_metrics(checkpoint['selected_metrics']['train'],actual)
            close_metrics(run_summary['selected_metrics']['train'],actual)
            for key in ('rmse','r2'):
                assert math.isclose(actual[key],row['train_'+key],rel_tol=1e-9,abs_tol=1e-8)
            if stage=='selection':
                initial_hashes.add(row['initial_model_sha256'])
                assert training['mode']=='selection' and training['max_epochs']==20 and training['patience']==4 and training['min_delta']==1e-4
                assert not (run/'test_metrics.json').exists()
                best=min(history,key=lambda epoch:(epoch['validation_rmse'],epoch['epoch']))
                assert best['epoch']==checkpoint['epoch']==row['best_epoch']
                stopper=EarlyStopper(4,1e-4)
                for index,epoch in enumerate(history):
                    stop=stopper.update(epoch['validation_rmse'])
                    assert not stop or index==len(history)-1
                assert len(history)==20 or stop
                measured=evaluate(model,make_loader(bundle.datasets['validation'],512,device='cuda'),checkpoint['task_object'],'cuda')
                close_metrics(checkpoint['selected_metrics']['validation'],measured)
                close_metrics(run_summary['selected_metrics']['validation'],measured)
                for key in ('rmse','r2'):
                    assert math.isclose(measured[key],row['validation_'+key],rel_tol=1e-9,abs_tol=1e-8)
                if row['lambda']==0:
                    original=torch.load(a1/'grid/h256_d04/training/best_model.pt',map_location='cpu',weights_only=True)
                    for key,value in original['model_state'].items():
                        torch.testing.assert_close(model.state_dict()[key].cpu(),value,rtol=0,atol=0)
            else:
                assert training['mode']=='fixed_refit' and training['max_epochs']==frozen['E_star']
                assert checkpoint['epoch']==row['checkpoint_epoch']==row['epochs_completed']==frozen['E_star']
                assert not run_summary['stopped_early'] and run_config['stopping_metric'] is None
                assert not any(key.startswith('validation_') for epoch in history for key in epoch)
                assert row['lambda']==chosen['lambda']
                base=next(item for item in baseline if item['seed']==row['seed'])
                assert row['baseline_initial_sha256']==base['initial_model_sha256']
                assert row['train_loader_generator_sha256']==base['train_loader_generator_sha256']
                test=json.loads((run/'test_metrics.json').read_text())
                assert test['checkpoint_epoch']==frozen['E_star'] and test['seed']==row['seed'] and test['test_assisted_stopping'] is False
                assert test['metrics']['count']==row['test_count']==26702
                for key in ('rmse','r2'):
                    assert test['metrics'][key]==row[key]
            audited.append({'stage':stage,'lambda':row['lambda'],'seed':row['seed'],'checkpoint_epoch':checkpoint['epoch'],
                            'restored_metrics_reproduced_on_cuda':True,'checkpoint':row['checkpoint'],
                            'checkpoint_sha256':sha256_file(output/row['checkpoint'])})
            print(f"Audited {stage}, lambda {row['lambda']:g}, seed {row['seed']}",flush=True)
            del model
    assert len(initial_hashes)==1
    table=pd.read_csv(output/'tables/summary_metrics.csv').set_index('model')
    for name,group in groups.items():
        for metric,prefix,suffix in (('rmse','test_rmse','_sgd'),('r2','test_r2','')):
            values=np.array([row[metric] for row in group['per_seed']])
            assert math.isclose(table.loc[name,prefix+'_mean'+suffix],values.mean(),rel_tol=1e-12)
            assert math.isclose(table.loc[name,prefix+'_sample_std'+suffix],values.std(ddof=1),rel_tol=1e-12)
    assert len(pd.read_csv(output/'tables/lambda_search.csv'))==5
    assert len(pd.read_csv(output/'tables/per_seed_comparison.csv'))==10
    assert len(pd.read_csv(output/'tables/paired_differences.csv'))==5
    for name in ('architecture','lambda_search','selected_learning_curves','baseline_comparison'):
        for extension in ('png','pdf'):
            assert (output/'figures'/f'{name}.{extension}').stat().st_size>1000
    setup=json.loads((ROOT/'artifacts/setup/environment_check.json').read_text())
    for name,original in setup['files'].items():
        assert sha256_file(ROOT/name)==original['sha256'],f'Supplied file changed: {name}'
    a1_audit=json.loads((a1/'verification.json').read_text())
    for row in a1_audit['checkpoints']:
        assert sha256_file(a1/f"grid/h{row['hidden_width']:03d}_d{row['embedding_dimension']:02d}/training/best_model.pt")==row['checkpoint_sha256']
    assert sha256_file(a1/'refit/best_model.pt')==a1_audit['refit_checkpoint_sha256']
    for row in json.loads((a2/'verification.json').read_text())['checkpoints']:
        assert sha256_file(a2/f"runs/{row['variant']}/seed_{row['seed']}/best_model.pt")==row['checkpoint_sha256']
    evidence={'status':'passed','checked_at_singapore':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        'run_directory':str(output),'search_checkpoints':5,'refit_checkpoints':5,'selected_lambda':chosen['lambda'],
        'all_10_restored_training_metrics_reproduced_on_cuda':True,'all_5_restored_validation_metrics_reproduced_on_cuda':True,
        'lambda_zero_trained_baseline_weights_match_a1_exactly':True,'validation_only_lambda_selection_verified':True,
        'fixed_refit_epochs_and_once_test_records_verified':True,'same_seed_initial_mlp_and_batch_order_verified':True,
        'preprocessing_and_source_provenance_verified':True,'sample_sd_and_paired_statistics_verified':True,
        'no_additional_test_performance_evaluation':True,'supplied_files_and_a1_a2_checkpoints_unchanged':True,
        'figures_and_tables_checked':True,'checkpoints':audited}
    evidence_dir=ROOT/'artifacts/a3_checks'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    evidence_dir.mkdir(parents=True,exist_ok=False)
    write_json(output/'verification.json',evidence)
    write_json(evidence_dir/'verification.json',evidence)
    write_json(ROOT/'artifacts/a3_checks/latest.json',{'status':'passed','directory':str(evidence_dir),'verification':str(evidence_dir/'verification.json')})
    print('A3 audit passed: 5 validation checkpoints, 5 fixed refits, selection/seed controls, and unchanged supplied/A1/A2 files.')


if __name__=='__main__':
    main()
