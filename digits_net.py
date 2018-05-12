import numpy as np
from layer import Layer
import digits_data_reader as reader
import actions


def get_sum_deltas(i, layer_deltas, layer_pers):
    weights_i = np.array([layer_pers[j].weights[i] for j in range(len(layer_pers))])
    return np.dot(weights_i, layer_deltas)


def get_deltas_from_prev_layer(size, prev_layer_deltas, prev_layer, layer):
    result = np.array(layer.last_result)
    vector = [get_sum_deltas(i, prev_layer_deltas, prev_layer.get_layer()) for i in range(size)]
    deltas = result * (1 - result) * np.array(vector)
    return deltas


class digitNetwork:
    def __init__(self):
        self.input_layer = Layer(784, actions.S, 1)
        self.hidden_layer_1 = Layer(16, actions.S, 784)
        self.hidden_layer_2 = Layer(16, actions.S, 16)
        self.output_layer = Layer(10, actions.S, 16)

    def get_output(self, data):

        input_result = self.input_layer.get_output_for_input_layer(data)
        hidden_result_1 = self.hidden_layer_1.get_output(input_result)
        hidden_result_2 = self.hidden_layer_2.get_output(hidden_result_1)
        output_result = self.output_layer.get_output(hidden_result_2)

        return output_result

    def learn(self, y, result):

        output_deltas = (result - y) * result * (1 - result)
        hidden_2_deltas = get_deltas_from_prev_layer(16, output_deltas, self.output_layer, self.hidden_layer_2)
        hidden_1_deltas = get_deltas_from_prev_layer(16, hidden_2_deltas, self.hidden_layer_2, self.hidden_layer_1)
        input_deltas = get_deltas_from_prev_layer(784, hidden_1_deltas, self.hidden_layer_1, self.input_layer)

        self.input_layer.learn_input(-input_deltas)
        self.hidden_layer_1.learn(-hidden_1_deltas)
        self.hidden_layer_2.learn(-hidden_2_deltas)
        self.output_layer.learn(-output_deltas)


def get_transformed_result(int):
    result = np.array(np.zeros(10))
    result[int] = 1
    return result


def learn(net, data):
    for i in range(2):
        counter = 0
        for row in range(len(data)):
            example = np.array(data[row, 1:]) / 255
            result = np.array(net.get_output(example.tolist()))
            correct = get_transformed_result(data[row][0])
            net.learn(correct, result)
            counter += 1
            if (counter % 1000 == 0):
                print(counter)



def get_int_from_result(result):
    max = result[0]
    index = 0
    for i in range(len(result)):
        if max < result[i]:
            max = result[i]
            index = i

    return index


def test(net, data):
    all = 0
    results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for row in range(len(data)):
        result = net.get_output((np.array(data[row][1:]) / 255).tolist())
        int = get_int_from_result(result)
        results[int] += 1
        if int == data[row][0]:
            correct[int] += 1
        all += 1
        if row % 1000 == 0:
            print(row)
    print((all, correct, results))


def save(network):
    pass


def main():
    net = digitNetwork()
    data = reader.get_data_from_file("digits_data/train.csv")
    learn(net, data)
    #data = reader.get_data_from_file("digits_data/test.csv")
    #print(data.shape)
    #return
    test(net, data)
    save(net)



if __name__ == "__main__":
    main()
