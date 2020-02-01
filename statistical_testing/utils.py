from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import IO
from configuration_files import setup_config
io_obj = IO()


class utils:
    def __init__(self, config):
        self.datalist = io_obj.load_datasets(config)

    def dataset_details(self):
        for key, value in self.datalist.items():
            for sub_key, sub_val in value.items():
                print(sub_key)

        return None


def main():
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs

    utl = utils(configuration)
    utl.dataset_details()


if __name__ == '__main__':
    main()
