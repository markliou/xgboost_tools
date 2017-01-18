#!/usr/bin/python3
#####
# This script handles the libsvm format file as the input, and just for binary classification
# For the detail about the libsvm format, please refer https://www.quora.com/What-is-this-data-format-in-LIBSVM-training-dataset
# markliou                     Licence: GPLv3
#####

import argparse
import xgboost as xb

### subroutine area start ###
def XGboost_grid(dtrain, num_boost_round = 100, early_stopping_rounds = 10, evalu = 'auc'):
    
    params_generator = xgboost_params_gen(evalu) # make the parameters
    
    # intial first result 
    params_c = next(params_generator)
    res = xb.cv(params_c,dtrain, nfold=args.nfold, metrics=[evalu, 'error'], early_stopping_rounds = early_stopping_rounds)
    best_perform = res['test-'+evalu+'-mean'][len(res['test-'+evalu+'-mean'])-1]
    best_acc = res['test-error-mean'][len(res['test-error-mean'])-1]
    best_param = params_c.copy()
    
    # enter into grid search
    for params_c in params_generator:
        res = xb.cv(params_c,dtrain, params_c['number_boost_round'], nfold=args.nfold, metrics=[evalu, 'error'], early_stopping_rounds = early_stopping_rounds)
        res_perform_c = res['test-'+evalu+'-mean'][len(res['test-'+evalu+'-mean'])-1]
        res_acc_c = res['test-error-mean'][len(res['test-error-mean'])-1]
        if(res_perform_c > best_perform):
            best_perform = res_perform_c
            best_acc = res_acc_c
            best_param = params_c.copy()
            next
        next
        #print(params_c, res_c, best_perform, best_param)
        print( 'current ACC: ' + str(1-res_acc_c) +', best ACC: '+ str(1-best_acc) )
        print('current params:',end='')
        print(params_c)
        print('best params:',end='')
        print(best_param)
        
    for i in range(1,10):
        print('=',end='')
    print('\nbest ACC:'+str(1-best_acc))
    print('params:',end='')
    print(best_param)
    
    next
    
def xgboost_params_gen(evalu):
    #booster = [ 'gbtree','gblinear','dart' ]
    booster = [ 'gbtree' ]
    eta = [ 2 ** i for i in range(-10,5) ]
    gamma = [ 0.01 * i for i in range(1,6) ]
    max_depth = [ i for i in range(3,15) ]
    #max_depth = [ 12 ]
    subsample = [ 0.1 * i for i in range(5,10) ]
    #subsample = [0.8]
    num_boost_round = [1000, 2500, 5000]
    #num_boost_round = [1000]
    
    # define the parameters
    params_container = [
    {
    'booster'        : booster_c ,
    'eta'            : eta_c ,
    'gamma'          : gamma_c ,
    'max_depth'      : max_depth_c ,
    'subsample'      : subsample_c ,
    'number_boost_round': num_boost_round_c,
    'eval_metrics': evalu,
    #'objective': 'multi:softmax', 
    'objective': 'binary:logistic', 
    #'num_class': 2 ,
    'silent': 1
    }
    for booster_c in booster
    for gamma_c in gamma
    for max_depth_c in max_depth
    for subsample_c in subsample
    for num_boost_round_c in num_boost_round
    for eta_c in eta
    ]
    
    for params in params_container :
        yield params
    next
### subroutine area end ###




### main body

### parsing the cmd args
cmdpar = argparse.ArgumentParser()
cmdpar.add_argument("-F", "--File", help=" indicate the training file")
#cmdpar.add_argument("-N", "--num_boost_round", type = int, default = 100 , help=" Set the boosting round number (default: 100)")
#cmdpar.add_argument("-E", "--early_stopping_rounds", type = int, default = 10 , help=" Set the boosting round number (default: 10)")
#cmdpar.add_argument("-P", "--eval", default = 'auc' , help=" Set the boosting round number (default: auc)")
cmdpar.add_argument("-V", "--nfold", default = 5, type = int , help=" Set the cross-valudation fold number (default: 5)")

args = cmdpar.parse_args()
#print(args.num_boost_round)
#XGboost_grid(xb.DMatrix(args.File), evalu = args.eval)
XGboost_grid(xb.DMatrix(args.File))



