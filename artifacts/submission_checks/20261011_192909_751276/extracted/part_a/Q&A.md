Q: For task A1, when retraining on train+validation, should the scaler/encoder statistics be refitted on the combined set or kept from the training split?
A: Need to refit.


Q:In the assignment briefing, we are required to use BatchNorm and SAME padding for all convolutional layers. From what I understand, SAME padding is only commonly used for layers where stride = 1. May I ask for convolutional layers where stride = 2 (in B3-ResNet architecture), do we use forced SAME padding (to make output layers the same size as input layers), or does SAME padding work differently in this case?
A: You are partially correct. However, SAME padding does not keep the output the same size when stride is 2. It pads the input so that the output size is approximately the input size divided by the stride. So a stride 2 layer halves the feature map, which is the downsampling we want in the ResNet. In PyTorch, padding="same" is not allowed when the stride is greater than 1. Instead, you can work around it by using padding = kernel_size // 2.


Q: for B3 (no hyperparameter search), should the final model also be retrained on train+validation for its best epoch count before testing, as for B1/B2? Thanks in advance
A: Yes, that's correct. B3 should follow the same training protocol as B1 and B2 so that we can compare all three models fairly.


Q: Hi may I clarify if the plots of RMSE and R2 required in parts A2 and A3 means plots of RMSE and R2 against epochs, or will bar graphs comparing the RMSE and R2 values suffice?
A: Hi, RMSE and R2 should be plotted against the epochs.Please note the following:
Plot the train / validation metrics against epochs, not the test metrics.
Plot the two ablations experiments separately.
Validation metrics are plotted against epochs because we can see how the model has progressed during training. The test metrics evaluate only the final trained model. For instance, we can infer whether the model has converged from such plots. These plots can also help us determine whether the model is underfitting or overfitting.
For the test RMSE / R2, you can either report them as a table and / or use a bar plot.


Q: Can I clarify for part B when we are told to evaluate the final model using the test, does this refer to the exact model(weights+architecture) choosen by hyperparameter search or a new model retrained on the train+validation using the exact hyperparameters found in the search?
A: It is a new model. After the hyperparameter search, retrain a fresh model from scratch on the combined training and validation data. Use the selected hyperparameters and train for the best epoch count found during the search. Then evaluate this retrained model on the test set once.


Q: Dear Professor, could you please clarify two details for Part B? In chestxray_binary_64.npz, does 0 denote healthy subjects and 1 denote pneumonia patients? Where should dropout (p = 0.5) be applied in the three architectures in Figure B? I interpret the final fc,1 layer as a linear output producing a logit for binary cross-entropy, without ReLU. Is this interpretation correct?
A: Hi,Yes, 0 denotes normal (healthy) cases, and 1 denotes pneumonia cases.
For B1, apply dropout (p = 0.5) after ReLU in the two hidden fully connected layers, fc,64 and fc,32. B2 does not use dropout. The caption’s reference to “all layers” is incorrect.
Your interpretation of fc,1 is correct for B1/B2: it outputs a raw logit, with no ReLU or dropout. Pass this directly to BCEWithLogitsLoss, without applying sigmoid beforehand.
Thank you for pointing out the ambiguity.