Professor:
"Please note the following:

Plot the train / validation metrics against epochs, not the test metrics.
Plot the two ablations experiments separately.
Validation metrics are plotted against epochs because we can see how the model has progressed during training. The test metrics evaluate only the final trained model. For instance, we can infer whether the model has converged from such plots. These plots can also help us determine whether the model is underfitting or overfitting.
For the test RMSE / R2, you can either report them as a table and / or use a bar plot."


Question by student: "In the assignment brief, it states we are to use early stopping throughout the assignment. In the last part of A1, are we supposed to use the Test partition as an indicator for early stopping?"

Answer by professor: "Correct."