import yaml


class LoadConfig:
    def __init__(self, config_indicator):
        if config_indicator == 1:
            _path = r'E:\\apply\\york\\project\\source\\configuration_files\\WPDP.yaml'
        if config_indicator == 2:
            _path = r'E:\\apply\\york\\project\\source\\configuration_files\\CPDP.yaml'
        with open(_path) as file:
            self.exp_configs = yaml.load(file, Loader=yaml.FullLoader)
        self._multiple_datasets = ""
        self._number_of_folds = 0
        self._cross_validation_type = 0
        self._iterations = 0
        self._dataset_first = True
        self._evaluation_measures = []
        self._save_predictions = True
        self._save_confusion_matrix = True
        self._defect_models = ""
        self._results_destination = ""

        for key in self.exp_configs:
            if key == 'results_destination':
                self.results_destination = self.exp_configs[key]
            if key == 'number_of_folds':
                self.number_of_folds = self.exp_configs[key]
            if key == "multiple_files":
                self.multiple_datasets = self.exp_configs[key]
            if key == "cross_validation_type":
                self.cross_validation_type = self.exp_configs[key]
            if key == "iterations":
                self.iterations = self.exp_configs[key]
            if key == "dataset_first":
                self.dataset_first = self.exp_configs[key]
            if key == "evaluation_measures":
                self.evaluation_measures = self.exp_configs[key]
            if key == "save_predictions":
                self.save_predictions = self.exp_configs[key]
            if key == "save_confusion_matrix":
                self.save_confusion_matrix = self.exp_configs[key]

    # ##################################properties################################
    @property
    def granularity(self):
        return self._granularity

    @property
    def results_destination(self):
        return self._results_destination

    @property
    def number_of_folds(self):
        return self._number_of_folds

    @property
    def multiple_datasets(self):
        return self._multiple_datasets

    @property
    def cross_validation_type(self):
        return self._cross_validation_type

    @property
    def iterations(self):
        return self._iterations

    @property
    def dataset_first(self):
        return self._dataset_first

    @property
    def evaluation_measures(self):
        return self._evaluation_measures

    @property
    def save_predictions(self):
        return self._save_predictions

    @property
    def save_confusion_matrix(self):
        return self._save_confusion_matrix

    @property
    def defect_models(self):
        return self._defect_models

    # ########################################################setters#############################

    @multiple_datasets.setter
    def multiple_datasets(self, file):
        self._multiple_datasets = file

    @results_destination.setter
    def results_destination(self, file):
        self._results_destination = file

    @number_of_folds.setter
    def number_of_folds(self, k):
        if k <= 0:
            raise ValueError('Number of folds must be greater or equal to 0!')
        else:
            self._number_of_folds = k

    @cross_validation_type.setter
    def cross_validation_type(self, cv):
        if cv == 1 or cv == 2 or cv == 3:
            self._cross_validation_type = cv
        else:
            raise ValueError("Please select 1 or 2 or 3. Check configuration file please!")

    @iterations.setter
    def iterations(self, itr):
        if itr < 0:
            raise ValueError("Please select a positive number!")
        if isinstance(itr, float) or isinstance(itr, str):
            raise ValueError("Please select an integer number!")
        else:
            self._iterations = itr

    @dataset_first.setter
    def dataset_first(self, status):
        if isinstance(status, bool):
            self._dataset_first = status
        else:
            raise ValueError("The status should be True or False!")

    @evaluation_measures.setter
    def evaluation_measures(self, list_of_measures):
        self._evaluation_measures = list_of_measures

    @save_predictions.setter
    def save_predictions(self, pred):
        if isinstance(pred, bool):
            self._save_predictions = pred
        else:
            raise ValueError("The status should be True or False!")

    @save_confusion_matrix.setter
    def save_confusion_matrix(self, confu):
        if isinstance(confu, bool):
            self._save_confusion_matrix = confu
        else:
            raise ValueError("The status should be True or False!")

    @defect_models.setter
    def defect_models(self, models):
        if not isinstance(models, list):
            raise ValueError("Please enter names of defect models correctly!")
        else:
            self._defect_models = models
