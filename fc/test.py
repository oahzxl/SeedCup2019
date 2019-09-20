import time
import tensorflow as tf

import infenence
import train
import Data



def evaluate(data):
    x = tf.placeholder(
        tf.float32,[None,infenence.INPUT_NODE],name="x-input"
    )
    y_ = tf.placeholder(
        tf.float32,[None,infenence.OUTPUT_NODE],name="y-input"
    )
    validate_feed = {x:data.test_x(), y_:data.test_y()}

    y = infenence.inference(x,None)

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(
        train.MOVING_AVERAGE_DECAY
    )
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                train.MODEL_SAVE_PATH
            )
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                print("After %s training step,validation accracy = %g"%(global_step,accuracy_score))
            else:
                print("No checkpoint file found ")
                return

def main():
    data = Data.data()
    evaluate(data)

if __name__ == "__main__":
    main()





