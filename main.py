# !/usr/bin/env python3

import argparse
import configparser
import sys
import os
import platform
from functools import partial



# modules
from PPIprophet import io_ as io
from PPIprophet import validate_input as validate
from PPIprophet import gen_feat_v4 as gen_feat
from PPIprophet import predict
from PPIprophet import preprocess
from PPIprophet import score
from PPIprophet import combine


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
    parser = ParserHelper(description='PPIprophet argument')
    os_s = get_os()
    boolean = ['True', 'False']
    parser.add_argument(
        '-db',
        help='ppi network in STRING format',
        dest='database',
        action='store',
        default=False,
    )
    parser.add_argument(
        '-fdr',
        help='global FDR threshold',
        dest='fdr',
        action='store',
        default=0.1,
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
        '-crapome',
        help='crapome file path',
        dest='crap',
        action='store',
        default="crapome.org.txt",
    )
    parser.add_argument(
        '-thres',
        help='frequency threshold for crapome',
        dest='thres',
        action='store',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '-skip',
        help='skip the preprocessing step',
        dest='skip',
        action='store',
        default='False',
        type=str
    )
    args = parser.parse_args()

    # create config file
    config = configparser.ConfigParser()
    config['GLOBAL'] = {
        'db': args.database,
        'sid': args.sample_ids,
        'fdr': args.fdr,
        'temp': r'./tmp',
        'out': args.out,
        'crapome': args.crap,
        'thresh': args.thres,
        'skip': args.skip,
    }

    # create config ini file for backup
    with open('ProphetConfig.conf', 'w') as conf:
        config.write(conf)
    return config


def preprocessing(infile, config):
    tmp_folder = io.file2folder(infile, prefix=config['GLOBAL']['temp'])
    preprocess.runner(infile, db=config['GLOBAL']['db'])
    gen_feat.runner(tmp_folder)
    predict.runner(tmp_folder)

def main():
    config = create_config()
    validate.InputTester(config['GLOBAL']['sid'], 'ids').test_file()
    files = io.read_sample_ids(config['GLOBAL']['sid'])
    files = [os.path.abspath(x) for x in files.keys()]
    if config['GLOBAL']['skip'] == 'False':
        [preprocessing(infile, config) for infile in files]
    combine.runner(
                tmp_=config['GLOBAL']['temp'],
                ids=config['GLOBAL']['sid'],
                outf=config['GLOBAL']['out'],
                fdr=config['GLOBAL']['fdr']
                )
    score.runner(
                outf=config['GLOBAL']['out'],
                tmp_=config['GLOBAL']['temp'],
                crapome=config['GLOBAL']['crapome'],
                thresh=config['GLOBAL']['thresh']
                )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
