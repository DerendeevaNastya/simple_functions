class Perseptron:
    def __init__(self, converter):
        self.dendrit_weight = [1, 1, 1]
        self.epsilon = 0.2
        self.input = []
        self.converter = converter
        self.axon = 0

    def get_output(self, input_data):
        result = 0
        self.input = input_data
        for i in range(0, len(input_data)):
            result += input_data[i]*self.dendrit_weight[i]
        result += self.dendrit_weight[-1]
        self.axon = self.converter(result)
        return result

    def correct_weight(self, correct_result):
        d = correct_result - self.axon
        for i in range(len(self.dendrit_weight)):
            delta = self.epsilon * d * self.input[i]# * self.dendrit_weight[i]
            self.dendrit_weight[i] += delta
            self.dendrit_weight[i] += -0.1 if self.dendrit_weight[i] == 0 else 0