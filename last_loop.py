
import numpy as np

############################

'''
num_examples = 10
num_layers = 6

for example in range(num_examples):
    for layer in range(min(example + 1, num_layers)):
        print ('%d: layer: %d example: %d' % (example, layer, example - layer))
    print ()
    
print ('-----')
print ()
    
for example in range(1, num_layers):
    for layer in range(example, num_layers):
        # print ('(%d, %d): %d' % (example, layer, num_examples - layer + example - 1))
        print ('%d: %d' % (layer, num_examples - layer + example - 1))
    print ()
'''

############################
'''
num_examples = 2
num_layers = 6

for example in range(num_examples):
    for layer in range(min(example + 1, num_layers)):
        print ('%d: layer: %d example: %d' % (example, layer, example - layer))
    print ()
    
print ('-----')
print ()
    
for example in range(1, num_layers):
    for layer in range(example, min(num_examples + example, num_layers)):
        print ('%d: %d' % (layer, num_examples - layer + example - 1))
    print ()
'''
############################

num_examples = 10
num_layers = 6

for example in range(num_examples):
    for layer in range(min(example + 1, num_layers)):
        print ('%d: layer: %d example: %d' % (example, layer, example - layer))
    print ()
    
print ('-----')
print ()
    
for last_example in range(1, num_layers):
    for layer in range(last_example, min(num_examples + last_example, num_layers)):
        example = num_examples - layer + last_example - 1
        print ('layer: %d example: %d' % (layer, example))
    print ()
    
############################
    
    
    
    
    
    
