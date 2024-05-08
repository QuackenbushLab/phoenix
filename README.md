# PHOENIX
PHOENIX is a tool to infer **biologically explainable** differential equations describing genome-wide regulatory dynamics

**Background:**  Models that are formulated as ordinary differential equations (ODEs) can accurately explain temporal gene expression patterns and promise to yield new insights into important cellular processes, disease progression, and intervention design. Learning such ODEs is challenging, since we want to predict the evolution of gene expression in a way that accurately encodes the causal gene-regulatory network (GRN) governing the dynamics and the nonlinear functional relationships between genes. Most widely used ODE estimation methods either impose too many parametric restrictions or are not guided by meaningful biological insights, both of which impedes scalability and/or explainability.

**Results:**  To overcome these limitations, we developed PHOENIX, a modeling framework based on neural ordinary differential equations (NeuralODEs) and Hill-Langmuir kinetics, that can flexibly incorporate prior domain knowledge and biological constraints to promote sparse, biologically interpretable representations of ODEs. We test the accuracy of PHOENIX in a series of *in silico* experiments, benchmarking it against several currently used tools for ODE estimation. We demonstrate PHOENIX's flexibility by studying oscillating expression data from synchronized yeast cells. We also assess its scalability by modelling genome-scale expression for breast cancer samples ordered in pseudotime, as well as expression from B-cells treated with Rituximab.

**Conclusions:** PHOENIX uses a combination of user-defined prior knowledge and functional forms from systems biology to encode key properties of the underlying GRN, and subsequently predicting expression patterns in a biologically explainable manner.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
