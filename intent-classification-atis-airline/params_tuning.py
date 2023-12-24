import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft
from tensorflow.keras import layers
from keras_tuner.engine import base_tuner
from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs


num_epochs = 5
feature, label = "message", "is_flight_intent"

# GPU / CPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define parameter tuning with early stopping callbacks
TunerResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any])
])
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy", min_delta=0,
    patience=2, verbose=1, mode="auto", baseline=None,
    restore_best_weights=True
)

# Create utilities function
def transform_name(key):
    return f"{key}_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_output, n_epochs=num_epochs, batch_size=16):
    transform_feature_spec = (
        tf_output.transformed_feature_spec().copy()
    )
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern, batch_size=batch_size,
        features=transform_feature_spec, reader=gzip_reader_fn,
        num_epochs=n_epochs, label_key=transform_name(label)
    )
    return dataset

def model_builder(hp, vectorizer_layer):

    ## Define parameter used for tuning model
    n_hidden_layers = hp.Choice("num_hidden_layers", values=[1, 3])
    embed_dims = hp.Int("embed_dims", min_value=32, max_value=128, step=4)
    lstm_units = hp.Int("lstm_units", min_value=16, max_value=64, step=4)
    dense_units = hp.Int("dense_units", min_value=16, max_value=64, step=4)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.25)
    lr_rate = hp.Choice("learning_rate", values=[0.001, 0.01])

    ## Dynamic model architecture (Bi-LSTM)
    inputs = tf.keras.Input(shape=(1,), name=transform_name(feature), dtype=tf.string)
    x = vectorizer_layer(inputs)
    x = layers.Embedding(input_dim=10_000, output_dim=embed_dims)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    for _ in range(n_hidden_layers):
        x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    ## Compiling model architecture
    model.compile(
        optimizer="adam", loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model

def tuner_fn(fn_args: FnArgs):

    ## Define splitted dataset transformed
    tf_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_output, num_epochs)
    eval_set = input_fn(fn_args.eval_files[0], tf_output, num_epochs)

    ## Define vectorization layers
    vectorizer_dataset = train_set.map(lambda f, _: f[transform_name(feature)])
    vectorizer_layer = layers.TextVectorization(
        max_tokens=1_000, output_mode="int", output_sequence_length=500
    )
    vectorizer_layer.adapt(vectorizer_dataset)

    ## Apply parameter tuning
    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp, vectorizer_layer),
        objective=kt.Objective('binary_accuracy', direction='max'),
        max_epochs=num_epochs,
        factor=2,
        directory=fn_args.working_dir,
        project_name="kt_hyperband",
    )

    return TunerResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping],
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
