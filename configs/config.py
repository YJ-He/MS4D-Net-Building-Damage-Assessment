from configparser import ConfigParser
import configparser

"""
    # section distinguish the upper and lower letters, but key and value do not.
    The information in the configuration always store the data in string format, it will have translation when reading.
    [DEFAULT]: the value in [DEFAULT] offer default_value to all sections, and it owns the highest priority. 
    get: getboolean() getint() getfloat()
    get方法 提供一个更复杂的界面 保持向后兼容性 可以回退关键字仅提供回退值
    fallback 回退值

    refer values in other sections
    interpolation method: configparser.BasicInterpolation()
                          configparser.ExtendedInterpolation() ${section:key}
    config.set
    config.write
"""


class MyConfiguration():
    def __init__(self, config_file=None):
        super(MyConfiguration, self).__init__()

        # ./ current directory
        if config_file is None:
            config_file = r'./configs/config.cfg'

        config = ConfigParser()
        # interpolation method
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(filenames=config_file)

        self.config = config
        self.config_path = config_file
        self.add_section = 'Additional'
        print("Loaded config file successfully ...")
        config.write(open(config_file, 'w'))

    def add_args(self, key, value):
        self.config.set(self.add_section, key, value)
        self.config.write(open(self.config_path, 'w'))

    def write_to_file(self, config_file_path):
        self.config.write(open(config_file_path, 'w'))

    # string int float boolean
    @property
    def test_dir(self):
        return self.config.get("Directory", "test_dir")

    @property
    def test_log_dir(self):
        return self.config.get("Directory", "test_log_dir")

    @property
    def root_dir(self):
        return self.config.get("Directory", "root_dir")

    @property
    def save_dir(self):
        return self.config.get("Directory", "save_dir")

    @property
    def log_dir(self):
        return self.config.get("Directory", "log_dir")

    @property
    def pred_dir(self):
        return self.config.get("Directory", "pred_dir")

    @property
    def trainset_dir(self):
        return self.config.get("Directory", "trainset_dir")

    @property
    def validset_dir(self):
        return self.config.get("Directory", "validset_dir")

    @property
    def testset_dir(self):
        return self.config.get("Directory", "testset_dir")

    @property
    def data_folder_name(self):
        return self.config.get("Directory", "data_folder_name")

    @property
    def target_folder_name(self):
        return self.config.get("Directory", "target_folder_name")

    @property
    def model_name(self):
        return self.config.get("Directory", "model_name")

    @property
    def batch_size(self):
        return self.config.getint("Data", "batch_size")

    @property
    def input_channel(self):
        return self.config.getint("Data", "input_channel")

    @property
    def nb_classes(self):
        return self.config.getint("Data", "nb_classes")

    @property
    def original_size(self):
        return self.config.getint("Data", "original_size")

    @property
    def cropped_size(self):
        return self.config.getint("Data", "cropped_size")

    @property
    def input_size(self):
        return self.config.getint("Data", "input_size")

    @property
    def eval_size(self):
        return self.config.getint("Data", "eval_size")

    @property
    def overlapped(self):
        return self.config.getint("Data", "overlapped")

    @property
    def step_x(self):
        return self.config.getint("Data", "step_x")

    @property
    def step_y(self):
        return self.config.getint("Data", "step_y")

    @property
    def use_gpu(self):
        return self.config.getboolean("General", "use_gpu")

    @property
    def device_id(self):
        return self.config.getint("General", "device_id")

    @property
    def random_seed(self):
        return self.config.getint("General", "random_seed")

    @property
    def num_workers(self):
        return self.config.getint("General", "num_workers")

    @property
    def lr_algorithm(self):
        return self.config.get("Optimizer", "lr_algorithm")

    @property
    def init_lr(self):
        return self.config.getfloat("Optimizer", "init_lr")

    @property
    def lr_decay(self):
        return self.config.getfloat("Optimizer", "lr_decay")

    @property
    def momentum(self):
        return self.config.getfloat("Optimizer", "momentum")

    @property
    def weight_decay(self):
        return self.config.getfloat("Optimizer", "weight_decay")

    @property
    def epsilon(self):
        return self.config.getfloat("Optimizer", "epsilon")

    @property
    def monitor(self):
        return self.config.get("Train", "monitor")

    @property
    def dis_period(self):
        return self.config.getint("Train", "dis_period")

    @property
    def init_algorithm(self):
        return self.config.get("Train", "init_algorithm")

    @property
    def loss(self):
        return self.config.get("Train", "loss")

    @property
    def use_seed(self):
        return self.config.getboolean("Train", "use_seed")

    @property
    def use_one_cycle_lr(self):
        return self.config.getboolean("Train", "use_one_cycle_lr")

    @property
    def gamma(self):
        return self.config.getfloat("Train", "gamma")

    @property
    def use_cutmix(self):
        return self.config.getboolean("Train", "use_cutmix")

    @property
    def early_stop(self):
        return self.config.getint("Train", "early_stop")

    @property
    def max_samples(self):
        return self.config.getint("Train", "max_samples")

    @property
    def warmup_period(self):
        return self.config.getint("Train", "warmup_period")

    @property
    def save_period(self):
        return self.config.getint("Train", "save_period")

    @property
    def epochs(self):
        return self.config.getint("Train", "epochs")


if __name__ == '__main__':
    config = MyConfiguration("config.cfg")
    config.config.set("Directory", "root_dir",
                      r"D:\test")
    print(config.root_dir)
