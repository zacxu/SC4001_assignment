"""Part B entry point with one unrestricted output logit per image."""

from assignment_common.training import BinaryClassificationTask, TrainingConfig, fit


def train_classifier(model, train_loader, *, config: TrainingConfig, run_directory,
                     validation_loader=None, threshold=0.5, preprocessing=None,
                     model_config=None, on_epoch=None):
    return fit(model, train_loader, task=BinaryClassificationTask(threshold), config=config,
               run_directory=run_directory, validation_loader=validation_loader,
               preprocessing=preprocessing, model_config=model_config, on_epoch=on_epoch)
