import yaml


class LoadConfig:
    def __init__(self, config_indicator):
        if config_indicator == 1:
            _path = r'E:\\apply\\york\\project\\source\\configuration_files\\WPDP.yaml'
        if config_indicator == 2:
            _path = r'E:\\apply\\york\\project\\source\\configuration_files\\CPDP.yaml'
        with open(_path) as file:
            self.exp_configs = yaml.load(file, Loader=yaml.FullLoader)
        self._single_dataset = ""
        self._granularity = ""
        self._multiple_datasets = ""
        self._experiment_mode = 0
        self._cross_validation_type = 0
        self._iterations = 0
        self._dataset_first = True
        self._evaluation_measures = []
        self._save_predictions = True
        self._save_confusion_matrix = True
        self._defect_models = ""

        for key in self.exp_configs:
            if key == "experiment_mode":
                self.experiment_mode = self.exp_configs[key]
            if key == "granularity":
                self.granularity = self.exp_configs[key]
            if key == "single_file":
                self.single_dataset = self.exp_configs[key]
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
    def single_dataset(self):
        return self._single_dataset

    @property
    def granularity(self):
        return self._granularity

    @property
    def multiple_datasets(self):
        return self._multiple_datasets

    @property
    def experiment_mode(self):
        return self._experiment_mode

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
    @experiment_mode.setter
    def experiment_mode(self, exp_mode):
        if exp_mode == 1 or exp_mode == 2:
            self._experiment_mode = exp_mode
        else:
            raise ValueError("Please select either explorer or experimenter by entering 1 or 2 respectively!")

    @granularity.setter
    def granularity(self, grl):
        grl_range = range(1, 6)
        if grl in grl_range:
            self._granularity = grl
        else:
            raise ValueError("Please select numbers from 1 to 6!")

    @single_dataset.setter
    def single_dataset(self, file):
        self._single_dataset = file

    @multiple_datasets.setter
    def multiple_datasets(self, file):
        self._multiple_datasets = file

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


