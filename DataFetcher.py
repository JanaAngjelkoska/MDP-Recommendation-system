import csv


def read_file(file_name):
    with open(file_name, newline='', encoding='utf-8') as doc:
        csv_reader = csv.reader(doc, delimiter=',')
        dataset = list(csv_reader)[1:]

        dataset_v2 = []
        for row in dataset:
            transformed = []
            for element in row[1:]:
                try:
                    transformed.append(float(element))
                except ValueError:
                    transformed.append(element)

            dataset_v2.append(transformed)

    return dataset_v2


def get_attributes(path):
    with open(path, newline='', encoding='utf-8') as doc:
        csv_reader = csv.reader(doc, delimiter=',')
        attributes = list(csv_reader)[0][1:]
    return attributes


# DATASET AND ATTRIBUTES
dataset_path = "C:\\Users\\janaa\\PycharmProjects\\MDP recommendation system\\Dataset\\data.csv"
dataset = read_file(dataset_path)
attributes = get_attributes(dataset_path)
