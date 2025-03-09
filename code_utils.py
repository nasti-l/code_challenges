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

"""
OddOccurrencesInArray
Find value that occurs in odd number of elements.
"""
def find_single_element(A):
    """Finds the element that occurs an odd number of times"""
    result = 0
    
    # XOR all elements together
    for num in A:
        result ^= num
    
    return result

"""
FrogJmp
Count minimal number of jumps from position X to Y.
"""

def count_jumps(X, Y, D):
    '''Counts the number of jumps D to reach X from Y'''
    if Y<=X:
        return 0
    dist = Y-X
    return round_up(dist/D)

def round_up(n: int) -> int:
    '''Rounds up a number'''
    return int(n) + (n % 1 > 0)

'''
PermMissingElem
Find the missing element in a given permutation.
'''

def find_missing_integer(A):
    '''Finds the missing element in a given permutation'''
    actual_sum = sum(A)
    expected_sum = get_sum_of_n_elements_in_arithmetic_series(n=len(A)+1)
    return expected_sum - actual_sum

def get_sum_of_n_elements_in_arithmetic_series(n:int, d:int = 1, a: int = 1)->int:
    '''Finds the sum of n elements in an arithmetic series'''
    return int((n/2)*(2*a + (n-1)*d))