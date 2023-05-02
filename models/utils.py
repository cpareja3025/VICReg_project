import yaml
import pickle
import sys


def load_config(path):
    '''
    returns:
        yaml - yaml object containing the config file
    '''
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)