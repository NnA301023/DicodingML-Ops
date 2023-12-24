import os
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_hub as hub
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs


FEATURE_KEY, LABEL_KEY = "message", "is_flight_intent"


# ----------- Mengubah Nama Fitur
def transformed_name(key):
  """Renaming transformed features"""
  return key + "_xf"


# ----------- Memuat data dalam format TFRecord.
def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


# ----------- Memuat transformed_feature dan membagi dalam beberapa batch
def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


# ---------- Arsitektur Model
def model_builder(vectorizer_layer, hp):
    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )

    x = vectorizer_layer(inputs)
    x = layers.Embedding(input_dim=5000, output_dim=hp["embed_dims"])(x)
    x = layers.Bidirectional(layers.LSTM(hp["lstm_units"]))(x)

    for _ in range(hp["num_hidden_layers"]):
        x = layers.Dense(hp["dense_units"], activation=tf.nn.relu)(x)
        x = layers.Dropout(hp["dropout_rate"])(x)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs = outputs)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    model.summary()
    
    return model


# ---------- Menjalankan tahapan preprocessing data pada raw request data.
def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)
        
    return serve_tf_examples_fn


# ----------- Run Training
def run_fn(fn_args: FnArgs):
    hp = fn_args.hyperparameters["values"]
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    
    # Define Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch")
    
    # Define Callback
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy",
                                                  min_delta=0,
                                                  patience=12,
                                                  verbose=0,
                                                  mode="auto",
                                                  baseline=None,
                                                  restore_best_weights=True)
    
    # Define Model Check Poin Callbacks
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, 
                                                          monitor='val_binary_accuracy', 
                                                          mode='max', 
                                                          verbose=1, 
                                                          save_best_only=True)
    
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, 
                         tf_transform_output, 
                         hp['tuner/epochs'])
    
    eval_set = input_fn(fn_args.eval_files, 
                        tf_transform_output, 
                        hp['tuner/epochs'])
    
    vectorizer_dataset = train_set.map(
        lambda f, l: f[transformed_name(FEATURE_KEY)])
    
    vectorizer_layer = layers.TextVectorization(
        max_tokens=5000,
        output_mode='int',
        output_sequence_length=500
    )
    vectorizer_layer.adapt(vectorizer_dataset)

    # Build the model
    model = model_builder(vectorizer_layer, hp)
    
    # Train the model
    model.fit(x = train_set,
              steps_per_epoch=fn_args.train_steps,
              validation_data=eval_set,
              validation_steps=fn_args.eval_steps,
              callbacks=[tensorboard_callback, 
                         early_stop, 
                         model_checkpoint],
              epochs=hp['tuner/epochs'],
              verbose=1
              )
    
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, 
                                                    tf_transform_output).get_concrete_function(tf.TensorSpec(shape=[None],
                                                                                                             dtype=tf.string,
                                                                                                             name='examples'))
                  }
    model.save(fn_args.serving_model_dir, 
               save_format='tf', 
               signatures=signatures)
