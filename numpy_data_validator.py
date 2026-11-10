import numpy as np

def ffill_1d(arr: np.ndarray) -> np.ndarray:
    '''
    params:
    arr: np.ndarray (nd == 1)

    return:
    np.ndarray - arr where nan elements have been forward filled
    '''
    nan_mask = ~np.isnan(arr)
    idx = np.where(nan_mask, np.arange(len(arr)), 0)
    arr_ffilled = arr[np.maximum.accumulate(idx)]

    return arr_ffilled

def return_counter(arr: np.ndarray) -> np.ndarray:
    '''
    params:
    arr: np.ndarray (nd == 1)
        arr must be valid: without nan elements

    return:
    np.ndarray - array of return values (difference beetwen days in percents)
        shorter than input arr for 1 element [0] -> i+1 day
    '''
    return (arr[1:] - arr[:-1]) / arr[:-1]

def sma_ratio(arr: np.ndarray, window: int) -> np.ndarray:
    '''
    params:
    arr: np.ndarray (nd == 1)
        arr must be valid: without nan elements
    window: int
        sma window

    return:
    np.ndarray - sma_ratio array has been made with cumsum with window from input 
    '''
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    sma = (cumsum[window:] - cumsum[:-window]) / window

    assert len(arr[window-1:]) == len(sma)
    return (arr[window-1:] / sma - 1)