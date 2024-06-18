def cal_f1_score(tp, fp, fn):
    failure = False
    if not isinstance(tp, int):
        failure = True
        print(f"tp must be int -- currently input {type(tp)}")
    if not isinstance(fp, int):
        failure = True
        print(f"fp must be int -- currently input {type(fp)}")
    if not isinstance(fn, int):
        failure = True
        print(f"fn must be int -- currently input {type(fn)}")
    if failure:
        return
    if tp*fp*fn == 0:
        print("tp and fb and fn must be greater than zero")
        return
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2 * (precision*recall)/(precision+recall)
    return f1_score