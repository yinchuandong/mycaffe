import numpy as np
import caffe

caffe_root = '/home/john/caffe/'
model_def = caffe_root + 'examples/myimagenet/deploy.prototxt'
model_weights = caffe_root + 'examples/myimagenet/caffenet_train_iter_33000.caffemodel'
caffe.set_device(0)
caffe.set_mode_gpu()


def main():
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(50,        # batch size
                              3,         # 3-channel (BGR) images
                              227, 227)  # image size is 227x227
    # image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')

    count = 0
    correct = 0
    fr = open('test.txt')
    for line in fr.readlines():
        arr = line.strip().split(' ')
        image = caffe.io.load_image('data/' + arr[0])
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        label = output_prob.argmax()
        count = count + 1
        if label == int(arr[1]):
            correct = correct + 1
        # print 'predicted class is:', label
        if count % 100 == 0:
            print '--count: %d  correct:%d  accuracy: %f' % (count, correct, float(correct) / count)
    fr.close()

    print 'count: %d  correct:%d  accuracy: %f' % (count, correct, float(correct) / count)

    return


if __name__ == '__main__':
    main()
