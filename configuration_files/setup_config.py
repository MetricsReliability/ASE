import yaml


class LoadConfig:
    def __init__(self, config_indicator):
        if config_indicator == 1:
            _path = r'E:\\apply\\york\\project\\source\\configuration_files\\WPDP.yaml'
        if config_indicator == 2:
            _path = r'E:\\apply\\york\\project\\source\\configuration_files\\CPDP.yaml'
        with open(_path) as file:
            self.exp_configs = yaml.load(file, Loader=yaml.FullLoader)
        self._file_level_data_address = ""
        self._change_level_data_address = ""
        self._number_of_folds = 0
        self._validation_type = 0
        self._iterations = 0
        self._evaluation_measures = []
        self._save_predictions = True
        self._save_confusion_matrix = True
        self._defect_models = ""
        self._file_level_WPDP_cross_validation_results_des = ""
        self._file_level_different_release_results_des = ""
        self._cross_validation_type = 0

        for key in self.exp_configs:
            if key == 'file_level_WPDP_cross_validation_results_des':
                self._file_level_WPDP_cross_validation_results_des = self.exp_configs[key]
            if key == 'file_level_different_release_results_des':
                self._file_level_different_release_results_des = self.exp_configs[key]
            if key == 'number_of_folds':
                self.number_of_folds = self.exp_configs[key]
            if key == "file_level_data_address":
                self.file_level_data_address = self.exp_configs[key]
            if key == "change_level_data_address":
                self.change_level_data_address = self.exp_configs[key]
            if key == "validation_type":
                self.validation_type = self.exp_configs[key]
            if key == "cross_validation_type":
                self._cross_validation_type = self.exp_configs[key]
            if key == "iterations":
                self.iterations = self.exp_configs[key]
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
    def file_level_WPDP_cross_validation_results_des(self):
        return self._file_level_WPDP_cross_validation_results_des

    @property
    def file_level_different_release_results_des(self):
        return self._file_level_different_release_results_des

    @property
    def number_of_folds(self):
        return self._number_of_folds

    @property
    def file_level_data_address(self):
        return self._file_level_data_address

    @property
    def change_level_data_address(self):
        return self._change_level_data_address

    @property
    def validation_type(self):
        return self._cross_validation_type

    @property
    def cross_validation_type(self):
        return self._cross_validation_type

    @property
    def iterations(self):
        return self._iterations

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

    @file_level_data_address.setter
    def file_level_data_address(self, file):
        self._file_level_data_address = file

    @change_level_data_address.setter
    def change_level_data_address(self, file):
        self._change_level_data_address = file

    @file_level_WPDP_cross_validation_results_des.setter
    def file_level_WPDP_cross_validation_results_des(self, file):
        self._file_level_WPDP_cross_validation_results_des = file

    @file_level_different_release_results_des.setter
    def file_level_different_release_results_des(self, file):
        self._file_level_different_release_results_des

    @number_of_folds.setter
    def number_of_folds(self, k):
        if k <= 0:
            raise ValueError('Number of folds must be greater or equal to 0!')
        else:
            self._number_of_folds = k

    @validation_type.setter
    def validation_type(self, cv):
        if cv == 1 or cv == 2 or cv == 3 or cv == 4:
            self._cross_validation_type = cv
        else:
            raise ValueError("Please select 1 or 2 or 3. Check configuration file please!")

    @cross_validation_type.setter
    def cross_validation_type(self, cross_type):
        self._cross_validation_type = cross_type

    @iterations.setter
    def iterations(self, itr):
        if itr < 0:
            raise ValueError("Please select a positive number!")
        if isinstance(itr, float) or isinstance(itr, str):
            raise ValueError("Please select an integer number!")
        else:
            self._iterations = itr

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
