# Utility functions

import tensorflow as tf
import h5py

def load_weights(model, weights_file):
    """
        Load weights from saved file based on layer name.
        This is temporay solution for the problem discussed in
        https://spectrum.chat/larq/general/testing-binary-models-with-latent-weights~d9987409-132c-4232-8370-0f706fdd50bd
    """

    with h5py.File(weights_file, "r") as w:
        for l in model.layers:
            for _w in l.trainable_weights:
                tf.keras.backend.set_value(_w, 
                                           w["{}/{}".format(l.name, _w.name)])
    