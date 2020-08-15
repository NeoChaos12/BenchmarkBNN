# BenchmarkBNN
A benchmarking tool written with the aim of evaluating the performance of various Bayesian Neural Networks (BNNs), or 
models which approximate one, as the surrogate function when used in a Bayesian Optimization loop on a benchmark 
objective chosen from the library of benchmarks available in HPOlib2.

## Terminology

**Target Model:** The BNN to be evaluated. Treated as a black-box.

**Benchmark:** The objective against which the performance of the target model is evaluated.

**Hyperparameters:** By default, hyperparameters refers to the various tune-able settings of BenchmarkBNN itself. This 
follows the convention that the global optimization loop of the benchmarking tool is not aware of the specifics of the 
target model (whether the target itself has hyper-parameters or not). Any deviations of this convention will be clearly 
mentioned.

## Target Models

These are technically any model with a python API exposing a number of specific functions that enable the target model 
to be autonomously called, trained and evaluated.