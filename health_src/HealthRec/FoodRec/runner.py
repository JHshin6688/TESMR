# coding: utf-8

"""
Main entry
##########################
"""
import sys
sys.path.append('../')
import os
import argparse

from FoodRec.utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SCHGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Foodcom', help='name of datasets, Allrecipes or Foodcom')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')

    # seed 인자도 추가
    parser.add_argument('--seed', '-s', type=int, default=42, help='random seed')

    
    config_dict = {
        'gpu_id': 6,
    }

    args, _ = parser.parse_known_args()
    
    # seed는 cmd를 통해서 설정
    config_dict['seed'] = [args.seed]
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)

