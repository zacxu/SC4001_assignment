"""Report assets calculated from saved A3 selection and five-seed results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2] / 'artifacts/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


def save_figure(figure, path):
    figure.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def baseline_comparison_limits(groups):
    """Padded zoom ranges covering every paired score and mean +/- SD endpoint."""
    limits = {}
    for metric, step in (('rmse', 5000), ('r2', .025)):
        values = [row[metric] for name in ('baseline', 'wide_deep')
                  for row in groups[name]['per_seed']]
        for name in ('baseline', 'wide_deep'):
            stats = groups[name]['metrics'][metric]
            values.extend((stats['mean'] - stats['sample_std'],
                           stats['mean'] + stats['sample_std']))
        low, high = min(values), max(values)
        padding = .08 * max(high - low, step)
        limits[metric] = (np.floor((low - padding) / step) * step,
                          np.ceil((high + padding) / step) * step)
    return limits


def export_a3(directory, *, refresh_only=False):
    output = Path(directory)
    summary = json.loads((output/'summary.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('A3 export requires completed results')
    grid = json.loads((output/'lambda_search.json').read_text())
    selected = summary['selected']
    groups = summary['variants']
    paired = summary['paired_differences']
    tables, figures = output/'tables', output/'figures'
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    if not refresh_only:
        pd.DataFrame(grid).to_csv(tables/'lambda_search.csv', index=False)
        pd.DataFrame([{'model': name, **row} for name, group in groups.items() for row in group['per_seed']]).to_csv(tables/'per_seed_comparison.csv', index=False)
        pd.DataFrame(paired['per_seed']).to_csv(tables/'paired_differences.csv', index=False)
    metric_rows=[]
    for name in ('baseline','wide_deep'):
        group=groups[name]
        metric_rows.append({'model': name, 'lambda': 0. if name=='baseline' else selected['lambda'],
            'parameters': group['per_seed'][0]['trainable_parameters'], 'seeds':5, 'epochs':summary['epoch_budget'],
            'test_rmse_mean_sgd':group['metrics']['rmse']['mean'], 'test_rmse_sample_std_sgd':group['metrics']['rmse']['sample_std'],
            'test_r2_mean':group['metrics']['r2']['mean'], 'test_r2_sample_std':group['metrics']['r2']['sample_std']})
    if not refresh_only:
        pd.DataFrame(metric_rows).to_csv(tables/'summary_metrics.csv', index=False)

    if not refresh_only:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for axis,key,ylabel in zip(axes,('validation_rmse','validation_r2'),('Best validation RMSE (SGD)','R² at selected validation checkpoint')):
            axis.plot([row['lambda'] for row in grid],[row[key] for row in grid],'o-',color='#377ab4')
            axis.scatter([selected['lambda']],[selected[key]],s=90,marker='*',color='#bd6628',zorder=4,label=f"Selected λ={selected['lambda']:g}")
            axis.set(xlabel='Fixed lambda',ylabel=ylabel,xticks=[row['lambda'] for row in grid])
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
            if key.endswith('rmse'):
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:,.0f}'))
        fig.suptitle('A3: lambda selection on 2021 validation data, seed 42')
        fig.tight_layout()
        save_figure(fig,figures/'lambda_search')

        fig,axes=plt.subplots(1,2,figsize=(10,4.4))
        controls=[(grid[0],'#447daa','λ=0 baseline control')]
        if selected['lambda']!=0:
            controls.append((selected,'#bc6d32',f"Selected λ={selected['lambda']:g}"))
        for row,color,label in controls:
            history=pd.read_csv(output/row['run_directory']/'history.csv')
            for axis,metric,ylabel in zip(axes,('rmse','r2'),('RMSE (SGD)','R²')):
                axis.plot(history['epoch'],history['train_'+metric],color=color,label=label+' train')
                axis.plot(history['epoch'],history['validation_'+metric],color=color,linestyle='--',label=label+' validation')
                axis.axvline(row['best_epoch'],color=color,alpha=.3,linestyle=':')
                axis.set(xlabel='Completed epoch',ylabel=ylabel)
                axis.grid(alpha=.2)
                axis.legend(fontsize=7)
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:,.0f}'))
        fig.suptitle('A3: training/validation curves; dotted lines mark saved epochs')
        fig.tight_layout()
        save_figure(fig,figures/'selected_learning_curves')

    y_limits = baseline_comparison_limits(groups)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    colors=plt.cm.viridis(np.linspace(.1,.85,5))
    for axis,metric,ylabel in zip(axes,('rmse','r2'),('Test RMSE (SGD) — lower is better','Test R² — higher is better')):
        means=[groups[name]['metrics'][metric]['mean'] for name in ('baseline','wide_deep')]
        stds=[groups[name]['metrics'][metric]['sample_std'] for name in ('baseline','wide_deep')]
        axis.errorbar([0,1],means,yerr=stds,fmt='D',color='black',capsize=6,zorder=4,label='Mean ± sample SD')
        for index,(base,wide) in enumerate(zip(groups['baseline']['per_seed'],groups['wide_deep']['per_seed'])):
            offset=(index-2)*.035
            axis.plot([offset,1+offset],[base[metric],wide[metric]],'o-',color=colors[index],linewidth=.9,markersize=4.5,label=f"Seed {base['seed']}")
        axis.set(xticks=[0,1],xticklabels=['A2 baseline (ReLU)\n7,065 parameters',f"Wide-and-deep, λ={selected['lambda']:g}\n{selected['trainable_parameters']:,} parameters"],ylabel=ylabel,xlim=(-.45,1.45))
        axis.grid(axis='y',alpha=.2)
        axis.set_ylim(*y_limits[metric])
        if metric=='rmse':
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:,.0f}'))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False,fontsize=8)
    fig.suptitle(f"A3: five matched seeds, fixed {summary['epoch_budget']}-epoch refits\n"
                 "Zoomed y axes; all paired scores and mean ± sample SD shown", fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.89))
    save_figure(fig,figures/'baseline_comparison')

    if not refresh_only:
        # Code-native architecture diagram, with the inverse transform displayed once.
        fig,axis=plt.subplots(figsize=(10,3.7))
        axis.set(xlim=(0,10),ylim=(0,4))
        axis.axis('off')
        boxes=[((1.6,2),'Four shared embeddings\n+ six continuous inputs\n22-dimensional vector'),
               ((5,3),'Linear(22,256) → LayerNorm\n→ ReLU → Linear(256,1)'),
               ((5,1),'Linear(22,1)\nMultiply by fixed λ'),
               ((8.4,2),'Add in standardized space\nThen mean + std × sum\nPredicted price (SGD)')]
        for position,label in boxes:
            axis.text(*position,label,ha='center',va='center',fontsize=9,bbox={'boxstyle':'round,pad=.6','facecolor':'#edf3f8','edgecolor':'#476780'})
        for start,end in [((3.1,2.25),(3.5,3)),((3.1,1.75),(3.9,1)),((6.6,3),(7.1,2.3)),((6.1,1),(7.1,1.7))]:
            axis.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#476780','lw':1.4})
        axis.set_title('Figure A implementation: shared input, weighted linear branch, single price transform')
        save_figure(fig,figures/'architecture')

    base=groups['baseline']['metrics']
    wide=groups['wide_deep']['metrics']
    delta=paired['metrics']['rmse_delta']['mean']
    improvements=sum(row['rmse_delta']<0 for row in paired['per_seed'])
    relative=abs(delta)/base['rmse']['mean']*100
    baseline_selection=grid[0]
    val_delta=selected['validation_rmse']-baseline_selection['validation_rmse']
    mechanism=("Validation selected the zero-weight control. The extra branch is inactive, so this selected model is functionally the baseline and does not test a benefit from an active wide branch."
               if selected['lambda']==0 else
               "The linear branch can directly fit additive associations in the shared encoded features while the MLP models nonlinear interactions. It also creates a shorter gradient path to the shared embeddings. These changes can affect optimization, but neither gradient magnitudes nor branch-specific causal effects were measured. The observed validation/test directions provide evidence about this training recipe, rather than proof of a particular mechanism.")
    direction=("improved" if delta<0 else "worsened" if delta>0 else "matched")
    test_interpretation=("The test mean moves in the direction predicted by a beneficial wide contribution, with substantial seed variability still visible."
                         if delta<0 else "The final test comparison does not show an average improvement from the validation-selected configuration." if delta>0 else "The final test results match the baseline under this comparison.")
    table='\n'.join(f"| {'A2 baseline (ReLU)' if row['model']=='baseline' else 'Selected wide-and-deep'} | {row['parameters']:,} | {row['test_rmse_mean_sgd']:,.2f} ± {row['test_rmse_sample_std_sgd']:,.2f} | {row['test_r2_mean']:.6f} ± {row['test_r2_sample_std']:.6f} |" for row in metric_rows)
    search_table='\n'.join(f"| {row['lambda']:g} | {row['best_epoch']} / {row['epochs_completed']} | {row['train_rmse']:,.2f} | {row['validation_rmse']:,.2f} | {row['validation_r2']:.6f} |" for row in grid)
    a1=summary['a1_individual_final']
    text=f"""# A3 results and discussion draft

All results come from five completed validation-search runs and five fresh combined-data refits.

## Architecture and controlled protocol

One shared concatenation of four embeddings (dimension 4 each) and six continuous variables feeds both branches. The deep branch is the selected A1 Linear(22,256) → LayerNorm → ReLU → Linear(256,1) network; the wide branch is a single Linear(22,1). Their standardized outputs are added as `deep + lambda * wide`; the target mean and standard deviation are applied once afterward. Lambda is a fixed scalar, not a learned parameter.

The baseline has 7,065 parameters. The linear branch adds 23, giving 7,088 allocated trainable parameters (0.33% more); at lambda 0 the added branch is inactive and only 7,065 participate in optimization. Constructing the MLP before the wide branch preserves same-seed baseline initialization. Lambda 0 exactly reproduced A1's selected architecture's CUDA training/validation metrics and checkpoint epoch.

Search uses only 2017–2020 training and 2021 validation, seed 42, the original selection preprocessing, Adam learning rate 0.01, batch size 512, maximum 20 epochs, patience 4, and minimum monitored improvement 1e-4. All five candidates start from identical weights and use same-seed shuffling. Select the lowest best validation RMSE, breaking exact ties by smaller lambda. The detailed brief's grid {{0, 0.25, 0.5, 1, 2}} takes precedence over the slide typo.

## Validation lambda selection

| Lambda | Saved / completed epoch | Train RMSE (SGD) | Validation RMSE (SGD) | Validation R² |
| --- | ---: | ---: | ---: | ---: |
{search_table}

Validation selected **lambda={selected['lambda']:g}**, RMSE **SGD {selected['validation_rmse']:,.2f}**, R² **{selected['validation_r2']:.6f}**, at epoch **{selected['best_epoch']}**. Its validation RMSE change from lambda 0 was **SGD {val_delta:+,.2f}**. The selected train/validation RMSE gap was **SGD {selected['validation_rmse']-selected['train_rmse']:,.2f}**; this chronological gap alone does not establish overfitting. `figures/lambda_search` shows both validation metrics, and `figures/selected_learning_curves` shows training/validation progression and saved epochs.

## Five-seed final test comparison

The selected configuration was initialized fresh for seeds 42–46, using the exact A1 refit preprocessing, 116,427 combined 2017–2021 rows, and fixed **E*={summary['epoch_budget']}** epochs. Each fourth-epoch model received one final evaluation on all 26,702 test rows from 2022. No validation/test monitoring loader was supplied during these final refits. The saved A2 five-seed baseline was reused without retraining or another test evaluation. Each seed's MLP initialization and training-row order matched its A2 baseline.

Values are mean ± **sample** SD (`ddof=1`) across five seeds, calculated from complete-partition metrics for each checkpoint.

| Model | Parameters | Test RMSE (SGD) | Test R² |
| --- | ---: | ---: | ---: |
{table}

The selected wide-and-deep model {direction} mean RMSE by **SGD {abs(delta):,.2f} ({relative:.2f}%)**. RMSE improved in **{improvements}/5** paired seeds. Paired wide-and-deep-minus-baseline changes were RMSE **{delta:+,.2f} ± {paired['metrics']['rmse_delta']['sample_std']:,.2f} SGD** and R² **{paired['metrics']['r2_delta']['mean']:+.6f} ± {paired['metrics']['r2_delta']['sample_std']:.6f}**. These standard deviations describe variation across seeds, rather than confidence intervals or statistical significance. `figures/baseline_comparison` presents both final metrics with matched-seed lines and mean/sample-SD markers. Its zoomed y axes span RMSE SGD {y_limits['rmse'][0]:,.0f}–{y_limits['rmse'][1]:,.0f} and R² {y_limits['r2'][0]:.3f}–{y_limits['r2'][1]:.3f}. Every paired score and mean ± sample-SD endpoint remains visible; paired points and error bars show these comparisons without zero-origin bars.

A1's individual seed-42 final score remains **RMSE SGD {a1['rmse']:,.2f}, R² {a1['r2']:.6f}**. That result used test-assisted stopping; it is separately identified rather than treated as a five-seed mean.

## Mechanism, evidence, and limitations

{mechanism}

{test_interpretation}

The selected checkpoint occurred at epoch {selected['best_epoch']}, compared with epoch {baseline_selection['best_epoch']} for lambda 0. Its training RMSE was SGD {selected['train_rmse']-baseline_selection['train_rmse']:+,.2f} different while validation RMSE changed by SGD {val_delta:+,.2f}. These observations are consistent with a change in optimization/generalization under the fixed recipe. They do not establish reliably faster convergence from a single selection seed. The absolute mean paired test-RMSE change was SGD {abs(delta):,.2f}, compared with paired sample SD SGD {paired['metrics']['rmse_delta']['sample_std']:,.2f}; the average gain is modest relative to the observed variation.

For any positive lambda, the multiplier can be absorbed into the unconstrained linear weights and bias: `W_effective=lambda*W`, `b_effective=lambda*b`. All positive lambda candidates therefore have the same function class, although their initial effective branch scales and finite-time optimization differ. There is no weight decay in the existing Adam trainer, and the baseline has no dropout. This rescaling equivalence is also verified with matched predictions in a behavioral test. The zero-weight case removes the active linear contribution.

Limitations include the 23 additional parameters when the branch is active, shared learned embeddings rather than fixed raw features, lambda selection on one seed and one validation year, and a short fixed final budget selected for A1. That budget inherits A1's professor-authorized test-assisted stopping, so the 2022 comparison is not fully independent of prior test use. Chronological price/feature-distribution changes may affect generalization. Test outcomes were not used to change lambda, training settings, or stopping epochs; a validation advantage can fail to transfer to 2022. Five seeds quantify descriptive variability and support no significance claim by themselves.

## Saved evidence and A4 handoff

- `lambda_search.json`, `selection.json`, and `tables/lambda_search.csv`: all five candidates and the validation-only choice.
- `search/lambda_*/`: configurations, original-selection preprocessing, epoch histories, and validation-selected checkpoints.
- `refit/seed_*/`: five fresh configurations, exact refit preprocessing, training-only histories, fourth-epoch checkpoints, and single final-test records.
- `tables/per_seed_comparison.csv`, `tables/summary_metrics.csv`, and `tables/paired_differences.csv`: ten comparison records, required sample-SD summaries, and paired differences.
- `figures/`: architecture, lambda search, selected learning curves, and five-seed baseline comparison, each as PNG and vector PDF.
- `baseline_parity.json`, `configuration.json`, `summary.json`, and `frozen_a3_settings.json`: baseline control, source/settings provenance, completed results, and A4 settings.

For A4, pair `refit/seed_42/best_model.pt` with A1's seed-42 `refit/best_model.pt`, using the same saved refit preprocessing. Both checkpoints correspond to the four-epoch budget. This step does not execute A4 attributions.
"""
    (output/'analysis.md').write_text(text,encoding='utf-8')
