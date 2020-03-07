import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def load_data(vectorizer):
    """
    This function loads the data, preprocesses it using a vectorizer, and splits
    the entire dataset randomly into 70% training, 15% validation, and 15% test
    examples.
    :param vectorizer: a matrix of token counts
    :type vectorizer: CountVectorizer
    :return: validation file, test file, train file with their corresponding labels
    :rtype: list[int]
    """
    merged_label_array = []
    merged_news_file = []
    fake_news = open("clean_fake.txt", "r")
    # read fake news line by line
    fake_news_line = fake_news.readline()
    merged_news_file.append(fake_news_line)
    while fake_news_line:
        fake_news_line = fake_news.readline()
        if fake_news_line:
            merged_news_file.append(fake_news_line)
        merged_label_array.append(0)

    real_news = open("clean_real.txt", "r")
    real_news_line = real_news.readline()
    merged_news_file.append(real_news_line)
    while real_news_line:
        real_news_line = real_news.readline()
        if real_news_line:
            merged_news_file.append(real_news_line)
        merged_label_array.append(1)
    merged_file_array = vectorizer.fit_transform(merged_news_file).toarray()

    # use train_test_split to split the data into a set of train file for 70% and a set of test file and
    # validation combined for 30%
    train_file_set, test_file_set, train_label_set, test_label_set = train_test_split(merged_file_array,
                                                                                      merged_label_array,
                                                                                      train_size=0.7, test_size=0.3)
    # use train_test_split to split the data into a set of validation file set and test file set, and each
    # of them occupy 50%
    validation_file_set, test_file_set, validation_label_set, test_label_set = train_test_split(test_file_set,
                                                                                                test_label_set,
                                                                                                train_size=0.5,
                                                                                                test_size=0.5)
    return train_file_set, test_file_set, validation_file_set, train_label_set, test_label_set, validation_label_set


def select_model(train_file_set, train_label_set, validation_file_set, validation_label_set):
    """
     This function trains the decision tree classifier using at least 5 different values of max_depth,
     as well as two different split criteria (information gain and Gini coefficient), evaluates the
    performance of each one on the validation set, and prints the resulting accuracies of each model.
    :param train_file_set:  the set of files need to be trained
    :type train_file_set:  list[int]
    :param train_label_set:  the set of labels for the corresponding file set
    :type train_label_set: list[int]
    :param validation_file_set: the set of validation files
    :type validation_file_set: list[int]
    :param validation_label_set: the set of labels for the corresponding file set
    :type validation_label_set: list[int]
    :return: the model containing two criteria
    :rtype: DecisionTreeClassifier
    """
    highest_accuracy = 0
    best_model = None
    # set two criteria as split_criteria and height
    for split_criteria in ['entropy', 'gini']:
        for max_depth in range(2, 7):
            # get the model
            model = DecisionTreeClassifier(criterion=split_criteria, max_depth=max_depth)
            model.fit(train_file_set, train_label_set)
            # get the predicted label set
            predicted_label_set = model.predict(validation_file_set)
            count_true_prediction = 0
            # calculate the accuracy of this model with the predicted label set and the actual label set
            for i in range(len(validation_label_set)):
                if int(validation_label_set[i]) == int(predicted_label_set[i]):
                    count_true_prediction += 1
            accurate_rate = count_true_prediction / len(validation_file_set)
            # print the accuracy for this model
            print(split_criteria, " with max depth ", max_depth, " has an accuracy of ", accurate_rate)
            # find the most accurate model and return it
            if accurate_rate >= highest_accuracy:
                highest_accuracy = accurate_rate
                best_model = model
    print(best_model)
    return best_model


def visualize_decision_tree(best_model, feature_names):
    """
    Visualize the decision tree.
    :param best_model: the most accurate model for the decision tree
    :type best_model: DecisionTreeClassifier
    :param feature_names: a list of feature names
    :type feature_names: list[str]
    :return: integer zero
    :rtype: int
    """
    export_graphviz(best_model,
                    out_file="decision_tree.dot",
                    max_depth=1,
                    feature_names=feature_names,
                    class_names=['fake', 'real'],
                    special_characters=True,
                    filled=True,
                    rounded=True)
    return 0


def get_feature_index(word, vectorizer):
    """
    Get the index with given feature name.
    :param word: a string representing the feature name
    :type word: str
    :param vectorizer: a matrix of token counts
    :type vectorizer: CountVectorizer
    :return: an integer representing the index of the feature name
    :rtype: int
    """
    return vectorizer.vocabulary_.get(word)


def compute_root_entropy(train_label_set):
    """
    Compute the root level's entropy.
    :param train_label_set: a set of labels for the corresponding root set
    :type train_label_set: a set of labels for the corresponding label set
    :return: an float indicating the entropy of the root level
    :rtype: float
    """
    total_entropy = 0
    # count the number of appearance for 0 and 1
    label_counters = Counter(train_label_set)
    # calculate entropy with the formula we know
    for label in label_counters:
        prob = label_counters[label] / len(train_label_set)
        total_entropy -= prob * math.log(prob)
    return total_entropy


def compute_leaves_entropy(train_file_set, train_label_set, feature_index):
    """
    Compute the level of entropy for the leave level.
    :param train_file_set: a set of train files
    :type train_file_set: list[int]
    :param train_label_set: a set of labels for the corresponding train set
    :type train_label_set: list[int]
    :param feature_index: a set of
    :type feature_index: an integer representing the index for the feature name
    :return: an float indicating the leave level's entropy
    :rtype: float
    """
    # set a counter to count the number of news containing the given word and are fake
    fake_containing = 0
    # set a counter to count the number of news do not contain the given word and are fake
    fake_not_containing = 0
    # set a counter to count the number of news containing the given word and are real
    real_containing = 0
    # set a counter to count the number of news do not contain the given word and are real
    real_not_containing = 0
    # loop through all news in train_file_set and check whether they contain the given word or not
    for index in range(len(train_file_set)):
        if train_file_set[index][feature_index] == 0:
            if train_label_set[index] == 0:
                fake_not_containing += 1
            else:
                real_not_containing += 1
        else:
            if train_label_set[index] == 0:
                real_containing += 1
            else:
                fake_containing += 1
    # calculate the total number of news containing the given word
    total_containing = fake_containing + real_containing
    # calculate the total number of news does not contain the given word
    total_not_containing = fake_not_containing + real_not_containing
    # calculate the probability that the news is fake if containing the given word
    prob_fake_with_given_word = fake_containing / total_containing
    # calculate the probability that the news is fake if containing the given word
    prob_real_with_given_word = real_containing / total_containing
    # calculate the probability that the file is fake if does not contain the given word
    prob_fake_without_given_word = fake_not_containing / total_not_containing
    # calculate the probability that the file is real if does not contain the given word
    prob_real_without_given_word = real_not_containing / total_not_containing
    # calculate the entropy for the situation that news containing the given word
    entropy_with_given_word = - prob_fake_with_given_word * math.log(
        prob_fake_with_given_word) - prob_real_with_given_word * math.log(prob_real_with_given_word)
    # calculate the entropy for the situation that the news do not contain the given news
    entropy_without_given_word = - prob_fake_without_given_word * math.log(
        prob_fake_without_given_word) - prob_real_without_given_word * math.log(prob_real_without_given_word)
    # then calculate the total entropy for the leave level
    total_entropy = (total_containing / len(train_file_set)) * entropy_with_given_word + (
                total_not_containing / len(train_file_set)) * entropy_without_given_word
    return total_entropy


def compute_information_gain(train_file_set, train_label_set, feature_index):
    """
    Compute the information gain for a certain word.(represented by feature index in this case).
    :param train_file_set: a set of train files
    :type train_file_set: list[int]
    :param train_label_set: a set of labels for corresponding train labels
    :type train_label_set: list[int]
    :param feature_index: a feature's index
    :type feature_index:  int
    :return: a float representing the information on this word
    :rtype: float
    """
    # we first calculate the root entropy, then calculate the leave entropy, and find the difference between
    # between these two floats as information gain
    root_entropy = compute_root_entropy(train_label_set)
    leaves_entropy = compute_leaves_entropy(train_file_set, train_label_set, feature_index)
    return root_entropy - leaves_entropy


if __name__ == '__main__':
    vectorizer = CountVectorizer()
    train_file_set, test_file_set, validation_file_set, train_label_set, test_label_set, validation_label_set = load_data(
        vectorizer)
    best_model = select_model(train_file_set, train_label_set, validation_file_set, validation_label_set)
    feature_names = vectorizer.get_feature_names()
    visualize_decision_tree(best_model, feature_names)
    words = ['the', 'donald', 'trumps']
    for word in words:
        feature_index = get_feature_index(word, vectorizer)
        information_gain = compute_information_gain(train_file_set, train_label_set, feature_index)
        print("For word ", word, " has information gain ", information_gain)
