# !/usr/bin/env python3

import argparse
import configparser
import sys
import glob
import os
from time import time
import platform
from functools import partial


# modules
from APprophet import io_ as io
from APprophet import exceptions as exceptions
from APprophet import validate_input as validate
from APprophet import generate_features as gen_feat
from APprophet import predict as predict
from APprophet import preprocess as preprocess
from APprophet import score as score


class ParserHelper(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def get_os():
    return platform.system()


def create_config():
    '''
    parse command line and create .ini file for configuration
    '''
    parser = ParserHelper(description='Protein Complex Prophet argument')
    os_s = get_os()
    boolean = ['True', 'False']
    parser.add_argument(
        '-db',
        help='ppi network in STRING format',
        dest='database',
        action='store',
        default='20190513_CORUMcoreComplexes.txt',
    )
    # maybe better to add function for generating a dummy sample id?
    parser.add_argument(
        '-sid',
        help='sample ids file',
        dest='sample_ids',
        default='sample_ids.txt',
        action='store',
    )
    parser.add_argument(
        '-out',
        help='output folder name',
        dest='out',
        default='Output',
        action='store',
    )
    parser.add_argument(
        '-fdr',
        help='false discovery rate for novel complexes',
        dest='fdr',
        action='store',
        default=0.5,
        type=float,
    )
    args = parser.parse_args()

    # create config file
    config = configparser.ConfigParser()
    config['GLOBAL'] = {
        'db': args.database,
        'sid': args.sample_ids,
        'temp': r'./tmp',
        'out': args.out
    }
    config['POSTPROCESS'] = {'fdr': args.fdr}

    # create config ini file for backup
    with open('ProphetConfig.conf', 'w') as conf:
        config.write(conf)
    return config


def main():
    config = create_config()
    # validate.InputTester(config['GLOBAL']['db'], 'db').test_file()
    validate.InputTester(config['GLOBAL']['sid'], 'ids').test_file()
    files = io.read_sample_ids(config['GLOBAL']['sid'])
    files = [os.path.abspath(x) for x in files.keys()]
    for infile in files:
        # validate.InputTester(infile, 'in').test_file()
        tmp_folder = io.file2folder(infile, prefix=config['GLOBAL']['temp'])
        # preprocess.runner(infile)
        # gen_feat.runner(tmp_folder)
        # predict.runner(tmp_folder)
    combined.runner(
                tmp_=config['GLOBAL']['temp'],
                ids=config['GLOBAL']['sid'],
                outf=config['GLOBAL']['out']
                )
    score.runner(tmp_=config['GLOBAL']['out'])



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
