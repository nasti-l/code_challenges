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


"""
TapeEquilibrium
Minimize the value |(A[0] + ... + A[P-1]) - (A[P] + ... + A[N-1])|.
"""

def find_better_split(A):
   '''Finds the minimum difference between two summed parts of an array'''
   p = 1
   l = len(A)
   right_sum = sum(A[p:])
   left_sum = A[0]
   min_diff = calc_diff(left_sum=left_sum, right_sum=right_sum, mid_val=0)
   while p < l-1:
       diff = calc_diff(left_sum=left_sum, right_sum=right_sum, mid_val=A[p])
       if diff < min_diff:
           min_diff = diff
       left_sum += A[p]  # Add current element to left sum
       right_sum -= A[p]  # Remove current element from right sum
       p+=1
   return min_diff

def calc_diff(left_sum: int, right_sum: int, mid_val: int) -> int:
    '''Calculates the difference between two parts of an array'''
    return abs(left_sum-right_sum+2*mid_val)

"""
FrogRiverOne
Find the earliest time when a frog can jump to the other side of a river.
"""

def find_the_moment_all_elements_appeared_at_least_once(X, A):
    """Finds the moment all elements appeared at least once, if they did"""
    sum = X
    missing_leaves = [1]*X
    for i in range(0,len(A)):
        if missing_leaves[A[i]-1]:
            sum-=1
            if sum == 0:
                return i
            missing_leaves[A[i]-1] = 0
    return -1

"""
PermCheck
Check whether array A is a permutation.
"""

def if_permutation(A):
    sum = len(A)
    max_num = sum
    elements = [1] * sum
    for i in range(0,max_num):
        if A[i] > max_num:
            return 0
        if elements[A[i]-1]:
            sum -= 1
            if sum == 0:
                return 1
            elements[A[i]-1] = 0
    return 0

''''
PassingCars
Count the number of passing cars on the road.
'''

def count_car_accidents(A):
    '''Counts the number of car accidents'''
    m = 0
    ttl = 0
    for n in A:
        if n == 0:
            m+=1
        if ttl >= 1_000_000_000:
            return -1
        ttl += n*m

    return ttl


'''
CountDiv
Compute number of integers divisible by k in range [a..b].
'''
def count_deviders(A, B, K):
    '''Counts the number of integers divisible by K in the range [A, B]'''
    next_divisible = A + (K - A%K) % K
    if next_divisible > B:
        return 0
    out_of_range_divisible = B+1 + (K - (B+1)%K) % K
    return (out_of_range_divisible-next_divisible)//K