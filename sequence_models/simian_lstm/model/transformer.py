from __future__ import annotations

import keras
from keras.layers import Dense, Input, Reshape, Dropout
from keras_nlp.layers import TransformerEncoder, TransformerDecoder, TokenAndPositionEmbedding

from simian_lstm.model.hyperparameters import HyperParameters


def build_model(params: HyperParameters, feature_size):
    encoder_input = Input(shape=(params.sequence_length,), dtype="int16", name="input_sequence")
    embedding = TokenAndPositionEmbedding(
        vocabulary_size=feature_size+1,
        sequence_length=params.sequence_length,
        embedding_dim=params.embedding,
        mask_zero=True,
    )
    encoder_input_embedded = embedding(encoder_input)

    x = encoder_input_embedded

    if params.transformer_encoder:
        for _ in range(params.transformer_layers):
            encoder = TransformerEncoder(
                intermediate_dim=params.transformer_dim,
                num_heads=params.transformer_heads,
                dropout=params.dropout,
                kernel_initializer=params.initializer_obj,
            )
            x = encoder(x)

    if params.transformer_decoder:
        decoder_context = x
        decoder_input_embedded = embedding(encoder_input)

        y = decoder_input_embedded

        for _ in range(params.transformer_layers):
            decoder = TransformerDecoder(
                intermediate_dim=params.transformer_dim,
                num_heads=params.transformer_heads,
                dropout=params.dropout,
                kernel_initializer=params.initializer_obj,
            )
            if params.transformer_encoder:
                y = decoder(y, decoder_context)
            else:
                y = decoder(y)

        pre_output = y
    else:
        pre_output = x

    if not params.output_sequences:
        pre_output = Reshape(target_shape=[params.sequence_length*params.embedding])(pre_output)

    pre_output = Dropout(params.dropout)(pre_output)
    output = Dense(feature_size, activation="softmax", kernel_initializer=params.initializer_obj)(pre_output)
    model = keras.Model(encoder_input, output)

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=params.optimizer_obj, metrics=["accuracy"]
    )
    return model
