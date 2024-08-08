def levenshtein_distance(s1, s2):
    '''
    This is the function to calculate the Levenshtein. 
    Input: 2 strings A and B 
    It returns the min of step to convert A to B thought out 3 methods (Delete/Add/Remove)
    '''
    m = len(s1)
    n = len(s2)
    # Step 1
    result = [[0] * (n + 1) for _ in range(m + 1)]
    # Step 2
    for i in range(m + 1):
        result[i][0] = i
    for j in range(n + 1):
        result[0][j] = j
    # Step 3
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            result[i][j] = min(result[i - 1][j] + 1, result[i]
                               [j - 1] + 1, result[i - 1][j - 1] + cost)
    return result[m][n]
