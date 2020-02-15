from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import IO
from configuration_files import setup_config

io_obj = IO()


class utils:
    def __init__(self, config):
        self.datalist = io_obj.load_datasets(config)


def main():
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs

    utl = utils(configuration)
    utl.dataset_details()


if __name__ == '__main__':
    main()
