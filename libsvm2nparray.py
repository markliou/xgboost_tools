#!/usr/bin/python3
import numpy as np
import re

def libsvm2npa(libsvmfilename):
    labels = []
    data = []
    max_index = 0
    instances = 0

    # open libsvm file
    try:
        libsvmfile = open(libsvmfilename,'r') 
        #print(libsvmfile.readline())
    except IOError as err:
        print(err)
        exit()
        
    # parsing libsvm files and save the information into lists
    for line in libsvmfile:
        instances = instances + 1
        line = line.rstrip('\n') # remove the line change sign
        elements = re.split('\s+',line)
        #print(elements)
        labels.append(elements[0]) # keep the labels
        data.append(elements[1:]) # keep the features
        #print(data)
        for element in elements:
            #print(element)
            feature = re.split(':',element)
            if(len(feature)) == 2 :
                #print(feature[0] + '%%' + feature[1])
                # record max index
                if max_index < int(feature[0]) :
                    max_index = int(feature[0])
                    pass
                pass
            pass
        pass
    libsvmfile.close

    # make np array for the input to xgboost according to the libsvm file parsing results
    data_np = np.zeros((instances , max_index))
    #print(data_np[0][1])
    ## fill data into np array
    instant_c = 0
    for element in data:
        #print(element)
        for element_c in element :
            feature = re.split(':',element_c)
            if len(feature) == 2 :
                #print(str(instant_c) + ' ' + str(feature[0]))
                data_np[instant_c][int(feature[0])-1] = float(feature[1])
                pass
            pass
        instant_c = instant_c + 1
        pass
    #print(data_np[0])
    
    return {'labels':np.array(labels), 
            'features':data_np, 
            'class_set':list(set(labels)), 
            'class_num':len(set(labels)), 
            'max_index':max_index, 
            'instance_num':instances
            }
    pass
    
if __name__ == '__main__' :    
    data = libsvm2npa('a1a2')
    print(data['class_num'])
    print(data['features'][:,[2,10]])
