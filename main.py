
from seed_definer import *
from learn import *
from test import *
from mydataset import *
from mydataset import *
from utils.parameter_loader import *
from utils.make_file_name import *
from utils.logger import *

from utils.import_libraries import *

def main(): #learn(学習)とtest(精度評価)を行うプログラム

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_name", default='param0', help="setting yaml file name")
    args = parser.parse_args()

    yaml_file = args.yaml_name + ".yaml"
    base_parameters_dir = "./parameters"

    setting_yaml_file = os.path.join(base_parameters_dir, yaml_file)
    parameters = Parameters(setting_yaml_file)

    set_random_seed(parameters.seed)

    file_name = make_file_name(parameters)
    os.makedirs("./logs", exist_ok=True)

    logger.info(file_name)
    logger.info('Loading data ...')

    learn(parameters)

    test(parameters)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    main()



    