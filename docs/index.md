Documentation page for the Deep learning based Model Discovery package DeepMoD. DeePyMoD is a PyTorch-based implementation of the DeepMoD algorithm for model discovery of PDEs and ODEs.[github.com/PhIMaL/DeePyMoD](https://github.com/PhIMaL/DeePyMoD). This work is based on two papers: The original DeepMoD paper  [arXiv:1904.09406](http://arxiv.org/abs/1904.09406), presenting the foundation of this neural network driven model discovery and a follow-up paper [arXiv:2011.04336](https://arxiv.org/abs/2011.04336) describing a modular plug and play framework. 

## Summary 
![Screenshot](figures/DeepMoD_logo.png)
DeepMoD is a modular model discovery framewrok aimed at discovering the ODE/PDE underlying a spatio-temporal dataset. Essentially the framework is comprised of four components: 

*   Function approximator, e.g. a neural network to represent the dataset, 
*   Function library on which the model discovery is performed, 
*   Constraint function that constrains the neural network with the obtained solution 
*   Sparsity selection algorithm. 



