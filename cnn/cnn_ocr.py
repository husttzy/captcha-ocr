# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random
from io import BytesIO
from captcha.image import ImageCaptcha
import logging
import argparse
import os

class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    buf = ""
    for i in range(4):
        buf += str(random.randint(0,9))
    return buf

def get_label(buf):
    a = [int(x) for x in buf]
    return np.array(a)

def gen_sample(captcha, width, height):
    num = gen_rand()
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    img = img.transpose(2, 0, 1)
    return (num, img)

class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        self.captcha = ImageCaptcha(fonts=['./data/Ubuntu-C.ttf'])

        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        
    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.captcha, self.width, self.height)
                data.append(img)
                label.append(get_label(num))

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_ocrnet():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu1, p=0.25);
    print dropout1.shape

    conv2 = mx.symbol.Convolution(data=dropout1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")
    dropout2 = mx.symbol.Dropout(data=relu2, p=0.25);

    print dropout2.shape
    conv3 = mx.symbol.Convolution(data=dropout2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")
    dropout3 = mx.symbol.Dropout(data=relu3, p=0.25);
    print dropout3.shape

    conv4 = mx.symbol.Convolution(data=dropout3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")
    print relu4.shape
   # dropout4 = mx.symbol.Dropout(data=relu4, p=0.25);    

    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 256)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim = 0)
    label = mx.symbol.transpose(data = label)
    label = mx.symbol.Reshape(data = label, target_shape = (0, ))
    return mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")


def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 4):
        ok = True
        for j in range(4):
            k = i * 4 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total

if __name__ == '__main__':
    network = get_ocrnet()
    parser = argparse.ArgumentParser(description='train an image classifer on ocr') 
    devs = [mx.gpu(0)]
    num_epoch = 5
    learning_rate = 0.0005
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = num_epoch,
                                 learning_rate = learning_rate,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)
    
    batch_size = 16
    data_train = OCRIter(640000, batch_size, 4, 30, 80)
    data_test = OCRIter(1000, batch_size, 4, 30, 80)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    log_file = "cnn_ocr_"+str(batch_size)+"shape0.25_"+str(num_epoch)+"a"+ str(learning_rate)    
    log_dir = "."
    log_file_full_name = os.path.join(log_dir, log_file)
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)   
 
    model.fit(X = data_train, eval_data = data_test, eval_metric = Accuracy, batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save("cnn-ocr")
