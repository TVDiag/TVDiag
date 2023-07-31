class Process:
    def __init__(self, config: dict):
        common_args = config['common_args']
        self.dataset_name = common_args['dataset_name']
        self.dataset_args = config['dataset'][self.dataset_name]

    def process(self, reconstruct=False):
        pass