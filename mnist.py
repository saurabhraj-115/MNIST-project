import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'






#to preprocess the data and feed it to the model

def preprocess_data(im,label):
    #inputs images and corresponding labels
    im = tf.cast(im, tf.float32)
    #normalise
    im = im/127.5
    #to zero centre the data, between -1 and 1
    im= im-1
    #flatten from NxN to 1xN(^2) values
    im= tf.reshape(im, [-1])
    #return the processed data
    return im, label
    #this function literally did nothing with the "label", but removing it breaks the map funstion
    # so leave it for now 



 #helper function to create the data pipeline to hanfle train as well test/validation data
def create_dataset_pipeline(data_tensor,is_train= True, num_threads=8, batch_size=32, prefetch_buffer= 100):
    #read the data from a tensor ( npy matrix)
    dataset = tf.data.Dataset.from_tensor_slices(data_tensor)

    if is_train:


    #shuffle dataset. repeat to use the dataset multiple times
        dataset = dataset.shuffle(buffer_size=60000).repeat()
    #map funstion to preprocess the data, it allows to run functions in parallel. To preprocess the data
    #on several threads in parallel
    dataset = dataset.map(preprocess_data, num_parallel_calls = num_threads)
    #feed data in batches
    dataset = dataset.batch(batch_size)
    #to prevent data preprocessing and i/p o/p bottlenecks we can buffer the data
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset



#data_tensor is an N dim array which has our entire dataset, options for multithreaded function and also 
#setting the batch size of the data and prefetch buffer to avoid bottlneck for the GPU
def data_layer():
    #to keep organised. Clean and nice comp graph
    with tf.variable_scope("data"):
        data_train, data_val = tf.keras.datasets.mnist.load_data()
        dataset_train = create_dataset_pipeline(data_train, is_train = True)
        dataset_val = create_dataset_pipeline(data_val, is_train= False, batch_size=1)
        # #read the data from a tensor ( npy matrix)
        # dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
        # #shuffle dataset. repeat to use the dataset multiple times
        # dataset = dataset.shuffle(buffer_size=60000).repeat()
        # #map funstion to preprocess the data, it allows to run functions in parallel. To preprocess the data
        # #on several threads in parallel
        # dataset = dataset.map(preprocess_data, num_parallel_calls = num_threads)
        # #feed data in batches
        # dataset = dataset.batch(batch_size)
        # #to prevent data preprocessing and i/p o/p bottlenecks we can buffer the data
        # dataset = dataset.prefetch(prefetch_buffer)
        # #iterator return us the next batch of data everytime we call it
        #iterator = dataset.make_one_shot_iterator()



        #iterator to switch back and forth betwen the taining and validation sets
        iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_val.output_shapes) 
        init_op_train = iterator.make_initializer(dataset_train)
        init_op_val = iterator.make_initializer(dataset_val)
    #return them
    return iterator, init_op_train , init_op_val


#define the model/graph functions
#we will use the tf.layers module (high level api predefined), additonally we may write it from 
#scratch also if we want. Num of classes in MNIST is 10.
def model(input_layer, num_classes = 10):
    with tf.variable_scope("model"):
        #fully connected layer of 512 hidden units which inputs the images coming from the iterator
        net = tf.layers.dense(input_layer, 512)
        #pass the output from a rectified function relu (non linearrity)
        net = tf.nn.relu(net)
        #output layer, another dense layer
        net = tf.layers.dense(net, num_classes)
    return net


#logits are the o/p we get from our model that we feed into our
#softmax function
def loss_functions(logits, labels, num_classes=10):
    with tf.variable_scope("loss"):
        #convert out labels into one hot array
        target_prob = tf.one_hot(labels , num_classes)
        #loss function
        total_loss = tf.losses.softmax_cross_entropy(target_prob, logits)
    return total_loss




#define the optimiser for the loss function we just created

def optimizer_func(total_loss,global_step,learning_rate = 0.1):
    with tf.variable_scope("optimizer"):
        #we will compute the gradient metrics in this function
        optimizer = tf. train.GradientDescentOptimizer(learning_rate=learning_rate)
        #we take a step into the direction where the optimiser makes the loss minimum
        optimizer = optimizer.minimize(total_loss, global_step = global_step)
    return optimizer

#performance metrics to keep a check on the model as time progresses

def performance_metric(logits, labels):
    with tf.variable_scope("performance_metric"):
        #comparing the one hot encoding to find the maximum activation
        preds = tf.argmax(logits, axis=1)        
        #typecast so that the groundtruth labels and predicted lkabels are of the same type
        labels = tf.cast(labels, tf.int64)
        #compare the two vectors and create a binary mask
        corrects = tf.equal(preds, labels)
        #convert this binary mask into a float array, so that the mean function work sproperly
        accuracy = tf.reduce_mean(tf.cast(corrects,tf.float32))
    return accuracy


def train():
    global_step = tf.Variable(1, dtype= tf.int32, trainable= False, name = "iter_number")



    #training graph
    iterator, init_op_train , init_op_val = data_layer()
    images, labels = iterator.get_next()
    logits = model(images)
    loss = loss_functions(logits, labels)
    optimizer = optimizer_func(loss, global_step)
    accuracy = performance_metric(logits, labels)




    #start training
    num_iter=18750    # 10 epochs, i.e. 60000 / 32 batches = 1875
    log_iter= 1875
    #validate at every pass
    val_iter= 1875


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op_train)

        streaming_loss=0
        streaming_accuracy=0
        

        for i in range(1,num_iter +1):
            _,loss_batch, acc_batch=sess.run([optimizer,loss, accuracy])
            streaming_loss+=loss_batch
            streaming_accuracy+= acc_batch

            
            
            if i % log_iter == 0:
                print("Iteration: {}, Streaming loss: {:.2f}, Streaming accuracy: {:.2f}".format(i,streaming_loss/log_iter, streaming_accuracy/log_iter) )
                streaming_loss =0
                streaming_accuracy=0
            
            
            
            #for the validation part for this epoch
            if i% val_iter == 0:
                sess.run(init_op_val)
                validation_accuracy = 0
                num_iter = 0
                while True:
                    #iterate till end
                    try:
                        acc_batch = sess.run(accuracy)
                        validation_accuracy += acc_batch
                        num_iter += 1
                    #till end of dataset
                    except tf.errors.OutOfRangeError:
                        validation_accuracy/=num_iter
                        print("Iteration: {}, Validation_accuracy: {:.2f}".format(i,validation_accuracy))
                        #before we break out we gotta switch back to the training dataset
                        sess.run(init_op_train)
                        break


if __name__ == "__main__":
    # data_train, data_val = tf.keras.datasets.mnist.load_data()
    #print(data_train[0].shape, data_train[1].shape, data_val[0].shape, data_val[1].shape)
    train()














  #simple example of a multilayer perceptron with validation. 