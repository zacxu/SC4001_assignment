"""Part A entry point into the shared, standardized-target regression trainer."""

from assignment_common.training import RegressionTask, TrainingConfig, fit


def train_regressor(model, train_loader, *, target_mean, target_std, config: TrainingConfig,
                    run_directory, validation_loader=None, test_monitor_loader=None,
                    preprocessing=None, model_config=None, on_epoch=None):
    return fit(model, train_loader, task=RegressionTask(target_mean, target_std), config=config,
               run_directory=run_directory, validation_loader=validation_loader,
               test_monitor_loader=test_monitor_loader, preprocessing=preprocessing,
               model_config=model_config, on_epoch=on_epoch)
