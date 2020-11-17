# Automatic Conclusion Generation using Neural Networks
This project contains two approaches to infer a conclusion from a set of premises. A detailed description of how the 
approaches work can be found here: https://tzoellner.de/BachelorThesis.pdf

## Related Work
We build our approach upon the work of the following authors:
- The Target Inference Approach by Alshomary et al. (2020) (https://webis.de/downloads/publications/papers/alshomary_2020a.pdf)
- The BART model by Lewis et al. (2019) (https://arxiv.org/abs/1910.13461)

## Requirements
We suggest the following packages to be installed:
- flair
- transformers

## Project structure
This Project contains three libraries:
1. __Target Identification__: This Project contains code to identify premise and conclusion targets, given a trained ```SequenceTagger```
2. __BARTConclusionGeneration__: This Project uses BART models to create Conclusions
3. __TargetStanceGeneration__: This Project uses our Target-Stance Approach to generate a conclusion.