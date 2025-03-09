"""
BinaryGap: Find longest sequence of zeros in binary representation of an integer.
"""

def get_binary(number:int)->str:
    '''Converts a number to binary string'''
    return bin(number)[2:]

def find_max_gap(bin_str: str)->int:
    '''Finds the maximum gap of zeros in a binary string'''
    if bin_str == '0':
        return 1
    elif bin_str == '1': 
        return 0 
    max_gap = 0
    current_gap = 0
    for n in bin_str:
        if n == '1':
            max_gap = current_gap if current_gap > max_gap else max_gap
            current_gap = 0
        else: 
            current_gap += 1
    return max_gap

'''
CyclicRotation
Rotate an array to the right by a given number of steps.
'''
def rotate_array(A, K):
    '''Rotates an array to the right by a given number of steps'''
    if not A or K == 0:
        return A
    l=len(A)
    if l==1:
        return A
    k = find_k(l=l, K=K)
    if k==0:
        return A
    return A[-k:]+A[:-k]

def find_k(l:int, K:int)->int:
    '''Finds the number of steps to rotate an array'''
    return K%l