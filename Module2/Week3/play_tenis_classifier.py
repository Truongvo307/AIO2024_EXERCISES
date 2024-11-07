import numpy as np


def create_train_data():
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'no'],
        ['Sunny', 'Hot', 'High', 'Strong', 'no'],
        ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'no'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'yes']
    ]

    return np.array(data)


def compute_prior_probability(data):
    y_unique = ['yes', 'no']
    prior_probability = np.zeros(len(y_unique))
    play_tennis = data[:, -1]
    total_count = play_tennis.shape[0]
    yes_count = np.sum(play_tennis == y_unique[0])
    no_count = np.sum(play_tennis == y_unique[1])
    p_yes = yes_count / total_count
    p_no = no_count / total_count
    prior_probability = [p_yes, p_no]
    return prior_probability


def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []
    for i in range(0, train_data.shape[1]-1):
        # Collect feature samples from the dataset
        x_unique = np.unique(train_data[:, i])
        print(x_unique)
        list_x_name.append(x_unique)
        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(y_unique)):
            for k in range(0, len(x_unique)):
                x_conditional_probability[j][k] = len(np.where(
                    (train_data[:, i] == x_unique[k]) & (train_data[:, -1] == y_unique[j]))[0]) / len(np.where(train_data[:, -1] == y_unique[j])[0])
        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name


def train_naive_bayes(train_data):
    prior_probability = compute_prior_probability(train_data)
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    return prior_probability, conditional_probability, list_x_name


def get_id(f_name, list_features):
    return np.where(list_features == f_name)[0][0]


def predict(X, list_name, prior_probability, conditional_probability):
    x1 = get_id(X[0], list_name[0])
    x2 = get_id(X[1], list_name[1])
    x3 = get_id(X[2], list_name[2])
    x4 = get_id(X[3], list_name[3])

    p0 = prior_probability[0] * conditional_probability[0][0, x1] * conditional_probability[1][0,
                                                                                               x2] * conditional_probability[2][0, x3] * conditional_probability[3][0, x4]
    p1 = prior_probability[1] * conditional_probability[0][1, x1] * conditional_probability[1][1,
                                                                                               x2] * conditional_probability[2][1, x3] * conditional_probability[3][1, x4]
    return 0 if p0 > p1 else 1


if __name__ == '__main__':

    train_data = create_train_data()
    print(train_data)
    # prior_probability = compute_prior_probability(train_data)
    # print("P( play tennis = No)", prior_probability[0])
    # print("P( play tennis = Yes)", prior_probability
    # [1])
    prior_probability, conditional_probability, list_x_name = train_naive_bayes(
        train_data)
    print(prior_probability)
    print(conditional_probability)
    print('-------------------')
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    print(conditional_probability)
    print(prior_probability)
    
    id = get_id('Sunny', list_x_name[0])
    print(conditional_probability[0][1][id])

    if predict(['Sunny', 'Hot', 'High', 'Weak'], list_x_name, prior_probability, conditional_probability):
        print("Ad should go!") 
    else:
        print("Ad should not go!")
