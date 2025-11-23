# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import ast

from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    
    # seed 인자도 추가
    parser.add_argument('--seed', '-s', type=int, default=42, help='random seed')
    parser.add_argument('--cl_temperature', '-t', type=str, default=None, help='temperature of cl loss. e.g. "[0.1,0.2]"')
    parser.add_argument('--cl_weight', '-w', type=str, default=None, help='weight of cl loss. e.g. "[0.1,0.2]"')
    parser.add_argument('--gpu_id', '-gpu', type=int, default=0, help='gpu id')
    
    args, _ = parser.parse_known_args()
    config_dict = {
        'gpu_id': args.gpu_id,
    }
    
    # seed는 cmd를 통해서 설정
    config_dict['seed'] = [args.seed]
    if args.cl_weight is not None:
        config_dict['cl_weight'] = ast.literal_eval(args.cl_weight)
    if args.cl_temperature is not None:
        config_dict['cl_temperature'] = ast.literal_eval(args.cl_temperature)
        
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)