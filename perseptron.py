import random


class Perseptron:
    def __init__(self, converter, dendrites_count):
        self.dendrites_count = dendrites_count
        self.dendrites_weight = [random.random() for _ in range(dendrites_count + 1)]
        self.epsilon = 0.2
        self.input = []
        self.converter = converter
        self.axon = 0

    def get_output(self, input_data):
        result = 0
        self.input = input_data + [1]
        for i in range(0, len(input_data)):
            result += input_data[i] * self.dendrites_weight[i]
        result += self.dendrites_weight[-1]
        print(result)
        result = self.converter(result)
        return result

    def correct_weight(self, correct_result, result):
        d = correct_result - result
        for i in range(self.dendrites_count):
            delta = self.epsilon * d * self.input[i]
            self.dendrites_weight[i] += delta
            self.dendrites_weight[i] += -0.1 if self.dendrites_weight[i] == 0 else 0