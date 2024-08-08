def sliding(num_list, sli_win):
    n = len(num_list)
    result = []
    for i in range(n - sli_win + 1):
        result.append(num_list[i:sli_win+i])
    return result


def max_list_sliding_window(num_list, sli_win):
    result = []
    for i in range((len(num_list) - sli_win)+1):
        result.append(max(num_list[i:sli_win+i]))
    return result
