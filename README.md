# Metrics Reliability Project

In this project, we are going to evaluate the reliability of file-level and change-level defect prediction by conducting exploratory analysis.

## Project's structure

This project contains five packages as follows:
- benchmarks: for performance evaluation  and validation
- configuration_files: storing designed experiment as configuration files 
- data_collection_manipulation: handling input and output operations on disk
- datasets: storing datasets for defect prediction
- output: saving results

### configuration_files
The most important parts of the project are located in this package. This package contains configuration files that are customizable according to the designated experiment. The description of each field is commented in the config file.

### benchmarks
This package contains a module named **__main__** containing two classes:
```
class PerformanceEvaluation:
    def __init__(self, configuration):
        self.config = configuration
        self.recall_flag = False
        self.precision_flag = False
        self.F1 = False
        self.ACC_flag = False
        self.MCC_flag = False
        
 class Benchmarks:
    def __init__(self, dataset, dataset_names, _configuration):
        self.dataset = dataset
        self.config = _configuration
        self.dataset_names = dataset_names
        self.model_holder = self.config['defect_models']
```
The first class deals with calculating the performance of defect prediction models based on evaluation criteria defined in the configuration file. The second class handles evaluation types specified by the user in the config file.

### data_collection_manipulation
This package contains a module namely **data_handler** containing two classess as follows:
```
class DataPreprocessing:
class IO:
```
The first class performs data preprocessing operations on defect datasets. For example, in the first function of this class, useless attributes are identified and removed. The second function gets the size of each metric for further usage. The second class deals with reading datasets and writing the results of the experiments in already defined addresses in the config file.

### datasets
This directory contains some sub-directories for datasets categorizes based on the level of defect prediction. 
### output
Similar to the dataset directory, this one also contains some sub-directories to store the results based on the type od defect prediction experiment defined in the config file.
