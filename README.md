# Git Hub Directory Normalizing Flow Experiments

## To Do
- Code debugging for image data
- Run experiments to analyze effects of different hyperparameters on normalizing flow convergence for image data
- Parameters single normalizing flow or multiple normalizing flows, complexity of the model, number of input samples, activation functions, and others
- Run on application of image superresolution.

# Milestones Achieved

- Code working for 2d Distributions and converging for all 4 distributions
- Run experiments to analyze effects of different hyperparameters on normalizing flow convergence
- Parameters single normalizing flow or multiple normalizing flows, complexity of the model(number of linear layers and dimension of intermediate layers), number of input samples, activation functions, and others

# Ablation Experiments

- [Changing Concatenation of Flows](sample-outputs/changingNOF.md)
- [Changing Complexity of flow models](sample-outputs/changingModel.md)
- [Changing Number of input Samples Provided](sample-outputs/changingSamples.md)
- [Changing Noise in input samples](sample-outputs/changingNoise.md)
- [Changing Activation](sample-outputs/changingAct.md)

# Sample Outputs
## Two Moon Distribution samples from flow till epoch=10000
![](sample-outputs/two-moons/image.png)
## Circles Distribution samples from flow till epoch=10000
![](sample-outputs/circles/image.png)
## Spiral Distribution samples from flow till epoch=10000
![](sample-outputs/spiral2/image.png)
## Spiral 4 Distribution samples from flow till epoch=10000
![](sample-outputs/spiral4/image.png)


