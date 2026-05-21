from __future__ import annotations

import keras
from keras.layers import LSTM, Dense, Input, Dropout, Embedding

from simian_lstm.model.hyperparameters import HyperParameters


def build_model(params: HyperParameters, feature_size):
    inputs = Input(shape=[params.sequence_length], dtype="int32", name="input_sequence")

    if params.embedding:
        if params.embedding == -1:
            embedding_size = params.layers[0]
        else:
            embedding_size = params.embedding

        features = Embedding(input_dim=feature_size, output_dim=embedding_size)(inputs)
    else:
        # from keras.layers import CategoryEncoding
        # features = CategoryEncoding(num_tokens=feature_size, output_mode="one_hot")(inputs)

        # from keras.layers import TimeDistributed
        # features = TimeDistributed(CategoryEncoding(num_tokens=feature_size, output_mode="one_hot"))(inputs)

        inputs = Input(shape=[params.sequence_length, feature_size], dtype="float32", name="input_sequence")
        features = inputs

    if params.lstm_dropout:
        lstm_dropout = params.dropout
    else:
        lstm_dropout = 0.0

    for layer in params.layers[:-1]:
        features = LSTM(
            layer,
            dropout=lstm_dropout,
            return_sequences=True,
            kernel_initializer=params.initializer_obj
        )(features)

    features = LSTM(
        params.layers[-1],
        dropout=lstm_dropout,
        return_sequences=False,
        kernel_initializer=params.initializer_obj
    )(features)

    if params.dropout:
        features = Dropout(params.dropout)(features)

    output = Dense(feature_size, activation="softmax", kernel_initializer=params.initializer_obj)(features)

    model = keras.Model(inputs, output)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=params.optimizer_obj, metrics=["accuracy"]
    )
    return model
