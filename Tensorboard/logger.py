import tensorflow as tf
import numpy as np
import scipy.misc

from io import BytesIO

class Logger():
    def __init__(self, log_dir) -> None:
        "Create a summary writer logging to log_dir."
        self.writer = tf.summary.FileWriter(log_dir)
        
    def scalar_summary(self, tag, value, step):
        "Log a scalar variable."
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)