import numpy as np
import h5py
from tqdm.notebook import tqdm
import pandas as pd


def to_observed_matrix(data_file,aux_file):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    num = len(data_file.keys())
    id_order = aux_file['planet_ID'].to_numpy()
    observed_spectrum = np.zeros((num,52,4))

    for idx, x in enumerate(id_order):
        current_planet_id = f'Planet_{x}'
        instrument_wlgrid = data_file[current_planet_id]['instrument_wlgrid'][:]
        instrument_spectrum = data_file[current_planet_id]['instrument_spectrum'][:]
        instrument_noise = data_file[current_planet_id]['instrument_noise'][:]
        instrument_wlwidth = data_file[current_planet_id]['instrument_width'][:]
        observed_spectrum[idx,:,:] = np.concatenate([instrument_wlgrid[...,np.newaxis],
                                            instrument_spectrum[...,np.newaxis],
                                            instrument_noise[...,np.newaxis],
                                            instrument_wlwidth[...,np.newaxis]],axis=-1)
    return observed_spectrum

def to_competition_format(tracedata_arr, weights_arr, name="submission.hdf5",entry_name = "Planet_public"):
    """convert input into competition format.
    we assume the test data is arranged in assending order of the planet ID.
    Args:
        tracedata_arr (array): Tracedata array, usually in the form of N x M x 7, where M is the number of tracedata, here we assume tracedata is of equal size. It does not have to be but you will need to craete an alternative function if the size is different. 
        weights_arr (array): Weights array, usually in the form of N x M, here we assumed the number of weights is of equal size, it should have the same size as the tracedata
    Returns:
        None
    """
    submit_file = name
    RT_submission = h5py.File(submit_file,'w')
    for n in range(len(tracedata_arr)):
        ## sanity check - samples count should be the same for both
        assert len(tracedata_arr[n]) == len(weights_arr[n])
        ## sanity check - weights must be able to sum to one.
        assert np.isclose(np.sum(weights_arr[n]),1)

        grp = RT_submission.create_group(entry_name+str(n+1))
        pl_id = grp.attrs['ID'] = n 
        tracedata = grp.create_dataset('tracedata',data=tracedata_arr[n])         
        weight_adjusted = weights_arr[n]

        weights = grp.create_dataset('weights',data=weight_adjusted)
    RT_submission.close()

def read_data(path):
    import h5py
    trace = h5py.File(path,"r")
    return trace


def to_matrix(data_file_gt, indices):
    
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    id_order = indices
    data_file_gt=h5py.File(data_file_gt,'r')
    num = len(data_file_gt.keys())
    list_trace_gt = []
    list_weight_gt = []
    list_trace_predicted = []
    list_weight_predicted = []

    for x in id_order:
        current_planet_id = f'Planet_train{int(x+1)}'
        trace_gt_planet = np.array(data_file_gt[current_planet_id]['tracedata'])
        trace_weight_planet = np.array(data_file_gt[current_planet_id]['weights'])
        list_trace_gt.append(trace_gt_planet)
        list_weight_gt.append(trace_weight_planet)

    data_file_gt.close()
    return list_trace_gt, list_weight_gt

def default_prior_bounds():
    """Prior bounds of each different molecules."""

    #### check here!!!!!!####
    Rp_range = [0.1, 3]
    T_range = [0,7000]
    gas1_range = [-12, -1]
    gas2_range = [-12, -1]
    gas3_range = [-12, -1]
    gas4_range = [-12, -1]
    gas5_range = [-12, -1]
    
    bounds_matrix = np.vstack([Rp_range,T_range,gas1_range,gas2_range,gas3_range,gas4_range,gas5_range])
    return bounds_matrix

def restrict_to_prior(arr, bounds_matrix):
    """Restrict any values within the array to the bounds given by a bounds_matrix.

    Args:
        arr (array): N-D array 
        bounds_matrix (array): an (N, 2) shaped matrix containing the min and max bounds , where N is the number of dimensions

    Returns:
        array: array with extremal values clipped. 
    """
    arr = np.clip(arr, bounds_matrix[:,0],bounds_matrix[:,1])
    return arr

def normalise_arr(arr, bounds_matrix, restrict = True):
    if restrict:
        arr = restrict_to_prior(arr, bounds_matrix)
    norm_arr = (arr - bounds_matrix[:,0])/(bounds_matrix[:,1]- bounds_matrix[:,0])
    return norm_arr

def preprocess_trace_for_posterior_loss(tr, weights, bounds):
    import nestle
    trace_resampled = nestle.resample_equal(tr,weights )
    trace = normalise_arr(trace_resampled, bounds )
    return trace 

def to_observed_matrix(data_file,aux_file):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    num = len(data_file.keys())
    id_order = aux_file['planet_ID'].to_numpy()
    observed_spectrum = np.zeros((num,52,4))

    for idx, x in enumerate(id_order):
        current_planet_id = f'Planet_{x}'
        instrument_wlgrid = data_file[current_planet_id]['instrument_wlgrid'][:]
        instrument_spectrum = data_file[current_planet_id]['instrument_spectrum'][:]
        instrument_noise = data_file[current_planet_id]['instrument_noise'][:]
        instrument_wlwidth = data_file[current_planet_id]['instrument_width'][:]
        observed_spectrum[idx,:,:] = np.concatenate([instrument_wlgrid[...,np.newaxis],
                                            instrument_spectrum[...,np.newaxis],
                                            instrument_noise[...,np.newaxis],
                                            instrument_wlwidth[...,np.newaxis]],axis=-1)
    return observed_spectrum



def compute_posterior_loss(tr1, weight1, tr2, weight2, bounds_matrix=None):
    from scipy import stats
    if bounds_matrix is None:
        bounds_matrix = default_prior_bounds()
    n_targets = tr1.shape[1]
    trace1 = preprocess_trace_for_posterior_loss(tr1, weight1, bounds_matrix)
    trace2 = preprocess_trace_for_posterior_loss(tr2, weight2, bounds_matrix)

    score_trace = []
    for t in range(0, n_targets):
        resampled_gt = np.resize(trace2[:,t], trace1[:,t].shape)
        metric_ks = stats.ks_2samp(trace1[:,t], resampled_gt)
        score_trace.append(1 - metric_ks.statistic)
    return np.array(score_trace)

