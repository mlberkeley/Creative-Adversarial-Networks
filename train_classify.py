import classifier
import tensorflow as tf

with tf.Session() as sess:
    classifier = classifier.Classifier(sess)
    classifier.train()    



