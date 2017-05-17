from array import *
import numpy as np
from flask import Flask, jsonify, render_template, request

from mnist import model

import bnn
print(bnn.available_params(bnn.NETWORK_LFC))

classifier = bnn.PynqBNN(network=bnn.NETWORK_LFC)

classifier.load_parameters("mnist")

image_file="/home/xilinx/bnn/data/image.images-idx3-ubyte"

def regression(input):
    y = classifier.inference(image_file)
    print("Result:", y )
    return y

# webapp
app = Flask(__name__)

@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = (255 - np.array(request.json, dtype=np.uint8)).reshape(1,784)


    # Setting up the header of the MNIST format file        
    hexval = "{0:#0{1}x}".format(1,6)
    header = array('B')
    header.extend([0,0,8,1,0,0])
    header.append(int('0x'+hexval[2:][:2],16))
    header.append(int('0x'+hexval[2:][2:],16))
    header.extend([0,0,0,28,0,0,0,28])
    header[3] = 3 # Changing MSB for image data (0x00000803)

    for i in range(0,784):
        header.append(input[0,i])

    output_file = open(image_file, 'wb')
    header.tofile(output_file)
    output_file.close()

    # Inference
    output = regression(input)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
