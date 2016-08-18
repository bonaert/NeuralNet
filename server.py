import random
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from code.myOcr import OCR
import numpy as np

HOST_NAME = 'localhost'
PORT_NUMBER = 8000
HIDDEN_NODE_COUNT = 15

print("Host: %s:%s" % (HOST_NAME, PORT_NUMBER))

# Load data samples and labels into matrix
data_matrices = np.loadtxt(open('data.csv', 'rb'), delimiter=',')
data_labels = np.loadtxt(open('dataLabels.csv', 'rb'))

print(data_matrices.size)
print(data_labels.size)

# Convert from numpy ndarrays to python lists
data_matrices = data_matrices.tolist()
data_labels = data_labels.tolist()

# def black_or_white(val):
#     if val <= 0.3:
#         return 0
#     else:
#         return 1
#     if val >= 1:
#         return 1
#     else:
#         return val
#
#
# def process(data_matrix):
#     return list(map(black_or_white, data_matrix))
#
#
def print_data_matrix(bw_data_matrix):
    print(len(bw_data_matrix))
    for i in range(20):
        data = bw_data_matrix[i:400+i:20]
        for elem in data:
            if elem < 0.3:
                print(' ', end='')
            else:
                print('X', end='')
        print()
# If a neural network file does not exist, train it using all 5000 existing data samples.
# Based on data collected from neural_network_design.py, 15 is the optimal number
# for hidden nodes
train_indices = list(range(5000))
random.shuffle(train_indices)
nn = OCR(hidden_layer_size=HIDDEN_NODE_COUNT, data=data_matrices, correct_digit_for_data=data_labels,
         training_indices=train_indices)


class JSONHandler(BaseHTTPRequestHandler):
    def do_POST(s):
        response_code = 200
        response = ""
        var_len = int(s.headers.get('Content-Length'))
        content = s.rfile.read(var_len).decode()
        print(content)
        payload = json.loads(content)

        if payload.get('train'):
            training_samples = payload['trainArray']
            for info in training_samples:
                input_data = info["y0"]
                correct_digit = int(info["label"])
                nn.train_sample(input_data, correct_digit)
            nn.save_data()
        elif payload.get('predict'):
            try:
                input_data = payload['image']
                print_data_matrix(input_data)
                response = {"type": "test", "result": nn.predict(input_data)}
                print(response)
            except:
                response_code = 500
        else:
            response_code = 400

        s.send_response(response_code)
        s.send_header("Content-type", "application/json")
        s.send_header("Access-Control-Allow-Origin", "*")
        s.end_headers()
        if response:
            bytes_response = json.dumps(response).encode(encoding='UTF-8')
            s.wfile.write(bytes_response)
        return


if __name__ == '__main__':
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()
