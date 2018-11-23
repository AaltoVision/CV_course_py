import numpy as np

def from_data_file():
    """ This function reads the data that we use in this demo."""
    data=dict()
    
    import scipy.io as sio
    data_file = sio.loadmat('./data/data_train.mat')
    data['train']=dict()
    data['train']['inputs'] = data_file['inputs']
    data['train']['target'] = data_file['target']
    
    data_file = sio.loadmat('./data/data_validation.mat')
    data['val']=dict()
    data['val']['inputs'] = data_file['inputs']
    data['val']['target'] = data_file['target']
    
    data_file = sio.loadmat('./data/data_test.mat')
    data['test']=dict()
    data['test']['inputs'] = data_file['inputs']
    data['test']['target'] = data_file['target']
    
    return data
    
def theta_to_model(theta):
    """ This function takes a model (or gradient) in the form of one long vector (maybe produced 
    by model_to_theta), and restores it to the structure format, i.e. with fields 
    .input_to_hid and .hid_to_class, both matrices. """
    n_hid = np.int(theta.shape[0] / (256+10.))
    ret=dict()
    ret['input_to_hid'] = np.reshape(np.ravel(theta)[0:256 * n_hid], (n_hid, 256), order='F')
    ret['hid_to_class'] = np.reshape(np.ravel(theta)[256 * n_hid:], (10, n_hid), order='F')
    return ret

def initial_model(n_hid):
    """ This function initialises model parameters. """
    n_params = (256 + 10) * n_hid
    as_row_vector = np.cos(np.arange(n_params))
    return theta_to_model(as_row_vector[:,np.newaxis] * 0.1)  # We don't use random initialization, for 
                                                # this assignment. This way, everybody will get the same result 

def model_to_theta(model):
    # This function takes a model (or gradient in model form), 
    # and turns it into one long vector. See also theta_to_model.
    input_to_hid = np.ravel(model['input_to_hid'], order='F')[:,np.newaxis]
    hid_to_class = np.ravel(model['hid_to_class'], order='F')[:,np.newaxis]
    return np.vstack((input_to_hid, hid_to_class))

def logistic(input):
    ret = 1 / (1 + np.exp(-input))
    return ret

def log_sum_exp_over_rows(a):
    # This computes log(sum(exp(a), 1)) in a numerically stable way
    maxs_small = np.max(a, 0)
    maxs_big = np.tile(maxs_small[np.newaxis,:], (a.shape[0], 1))
    
    #print('maxs_small = {}'.format(maxs_small.shape))
    #print('maxs_big = {}'.format(maxs_big.shape))
    #print('a = {}'.format(a.shape))
    
    ret = np.log(np.sum(np.exp(a - maxs_big), 0)) + maxs_small
    
    return ret

def classification_performance(model, data):
    # This returns the fraction of data cases that is incorrectly classified by the model.
    hid_input = np.dot(model['input_to_hid'], data['inputs']) # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input) # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    class_input = np.dot(model['hid_to_class'], hid_output) # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
    choices = np.argmax(class_input,0) # choices is integer: the chosen class, plus 1.
    targets = np.argmax(data['target'],0) # targets is integer: the target class, plus 1.
    ret = np.mean((choices != targets).astype(np.float))
    return ret
