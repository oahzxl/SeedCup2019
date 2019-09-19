import os
import  tensorflow as tf
import Data
import infenence
num_examples = int(373877/5)*4
BATCH_SIZE = 1024
LEARNING_RATE_BASE = 0.000001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE  = 0.0001
TRAINING_STEPS  = 3000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"

def train(data):
    #定义输入输出
    x = tf.placeholder(
        tf.float32,[None,infenence.INPUT_NODE],name="x-input"
    )
    y_ = tf.placeholder(
        tf.float32,[None,infenence.OUTPUT_NODE],name = "y-input"
    )

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = infenence.inference(x,regularizer)
    global_step = tf.Variable(0,trainable = False)

    variables_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step
    )
    variables_averages_op = variables_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits= y,labels=tf.argmax(y_,1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    #learning_rate = LEARNING_RATE_BASE
    train_step = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

        for i in range(TRAINING_STEPS):
            xs,ys = data.next_batch(BATCH_SIZE,i)
            xs = xs.reshape([BATCH_SIZE,infenence.INPUT_NODE])
            #ys = ys.reshape([BATCH_SIZE,1])
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict = {x:xs,y_:ys})

            if i % 10 == 0:
                print("After %d training steps,loss on training batch is %d."%(step,loss_value))
                saver.save(
                    sess, MODEL_SAVE_PATH+"/"+MODEL_NAME,
                    global_step = global_step
                )

def main():
    data = Data.data("data.txt")
    train(data)

if __name__ == "__main__":
    main()