# -*- coding: utf-8 -*-
"""pipeline_tfx.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f3N2L5OVAKyPrAgfHJN3RIsegNWvbimX

### Intent Classification ATIS Airline

#### 0. Setup Environment
"""

# Install libraries
!pip install tfx tensorflow_model_analysis -q

# Load libraries
import os
import pandas as pd
from tfx.types import Channel
import tensorflow_model_analysis as tf_ma
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.components import (
    Transform, Trainer, Tuner, Evaluator, Pusher,
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator
    )

# # Reset pipeline created *Optional
# %rm -rf "/content/nna_alif-pipeline"

# Set global variable pipeline
PIPELINE_NAME = "intent-pipeline"
SCHEMA_NAME = "intent-tfdv-schema"
PIPELINE_ROOT = os.path.join("nna_alif-pipeline", PIPELINE_NAME) # create pipeline according format required 'dicoding_username-pipeline'
METADATA_PATH = os.path.join("metadata", PIPELINE_NAME, "metadata.db")
SERVING_MODEL_DIR = os.path.join("serving_model", PIPELINE_NAME)

DATA_ROOT = "/content/dataset"
INTERACTIVE_CONTEXT = InteractiveContext(pipeline_root=PIPELINE_ROOT)

# # Read dataset
# # NOTE: Because our dataset didnt have any header, we should define it first,
# #       else it might caused blank on schema define
# path = "https://raw.githubusercontent.com/NnA301023/DicodingML-Ops/main/intent-classification-atis-airline/dataset/atis_intents.csv"
# output_path = "/content/dataset/atis_intents_preprocessed.csv"
# data = pd.read_csv(path)

# # Overview dataset
# data.columns = ['intent', 'message']

# # Create binary label based on actual 'intent' label
# data['is_flight_intent'] = data['intent'].apply(lambda i: 1 if 'atis_flight' in i else 0)
# data['is_flight_intent'].value_counts()
# data = data[['message', 'is_flight_intent']]

# # Save .csv file
# data.to_csv(output_path, index=False)

# # Overview dataset
# data.head()

# Check uploaded dataset
!ls dataset

"""#### 1. Data Ingest"""

# Create data ingestion configuration using local .csv dataset
output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
        splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ]
    )
)
example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)

# Running data ingestion in InteractiveContext
INTERACTIVE_CONTEXT.run(example_gen)

"""#### 2. Data Validation

#####  2.1. Statistics Summaries
"""

# Create statictics information configuration on pipeline
statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

# Running data ingestion in InteractiveContext
INTERACTIVE_CONTEXT.run(statistics_gen)

# Overview statistics information in pipeline
INTERACTIVE_CONTEXT.show(statistics_gen.outputs["statistics"])

"""#####  2.2. Data Schema"""

statistics_gen.outputs["statistics"]

# Create schema dataset configuration
schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

# Execute dataset schema on pipeline
INTERACTIVE_CONTEXT.run(schema_gen)

# Overview schema result in pipeline
INTERACTIVE_CONTEXT.show(schema_gen.outputs["schema"])

"""##### 2.3. Data Anomaly Detection"""

# Create dataset anomaly detection configuration
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs["statistics"],
    schema=schema_gen.outputs["schema"]
)

# Execute anomaly detection in pipeline
INTERACTIVE_CONTEXT.run(example_validator)

# Menampilkan hasil dari validasi
INTERACTIVE_CONTEXT.show(example_validator.outputs['anomalies'])

"""#### 3. Data Preprocessing"""

# Create file transformation
TRANSFORM_FILE = "transforms.py"

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRANSFORM_FILE}
# import tensorflow as tf
# feature, label = 'message', 'is_flight_intent'
# 
# def transform_name(key: str) -> str:
#     return f"{key}_xf"
# 
# def preprocessing_fn(inputs):
#     outputs = {
#         transform_name(feature): tf.strings.lower(inputs[feature]),
#         transform_name(label): tf.cast(inputs[label], tf.int64)
#     }
#     return outputs

# Define transform component
transform = Transform(
    examples=example_gen.outputs["examples"],
    schema=schema_gen.outputs["schema"],
    module_file=os.path.abspath(TRANSFORM_FILE)
)

# Execute transform component in pipeline
INTERACTIVE_CONTEXT.run(transform)

"""#### 4. Model Development

#### 4.1. Tuning Parameters
"""

# Create file tuning parameters
TUNING_FILE = "params_tuning.py"

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TUNING_FILE}
# import tensorflow as tf
# import keras_tuner as kt
# import tensorflow_transform as tft
# from tensorflow.keras import layers
# from keras_tuner.engine import base_tuner
# from typing import NamedTuple, Dict, Text, Any
# from tfx.components.trainer.fn_args_utils import FnArgs
# 
# 
# num_epochs = 5
# feature, label = "message", "is_flight_intent"
# 
# # GPU / CPU Configuration
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# 
# # Define parameter tuning with early stopping callbacks
# TunerResult = NamedTuple("TunerFnResult", [
#     ("tuner", base_tuner.BaseTuner),
#     ("fit_kwargs", Dict[Text, Any])
# ])
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_binary_accuracy", min_delta=0,
#     patience=2, verbose=1, mode="auto", baseline=None,
#     restore_best_weights=True
# )
# 
# # Create utilities function
# def transform_name(key):
#     return f"{key}_xf"
# 
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
# 
# def input_fn(file_pattern, tf_output, n_epochs=num_epochs, batch_size=16):
#     transform_feature_spec = (
#         tf_output.transformed_feature_spec().copy()
#     )
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern, batch_size=batch_size,
#         features=transform_feature_spec, reader=gzip_reader_fn,
#         num_epochs=n_epochs, label_key=transform_name(label)
#     )
#     return dataset
# 
# def model_builder(hp, vectorizer_layer):
# 
#     ## Define parameter used for tuning model
#     n_hidden_layers = hp.Choice("num_hidden_layers", values=[1, 3])
#     embed_dims = hp.Int("embed_dims", min_value=32, max_value=128, step=4)
#     lstm_units = hp.Int("lstm_units", min_value=16, max_value=64, step=4)
#     dense_units = hp.Int("dense_units", min_value=16, max_value=64, step=4)
#     dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.25)
#     lr_rate = hp.Choice("learning_rate", values=[0.001, 0.01])
# 
#     ## Dynamic model architecture (Bi-LSTM)
#     inputs = tf.keras.Input(shape=(1,), name=transform_name(feature), dtype=tf.string)
#     x = vectorizer_layer(inputs)
#     x = layers.Embedding(input_dim=10_000, output_dim=embed_dims)(x)
#     x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
#     for _ in range(n_hidden_layers):
#         x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
#         x = layers.Dropout(dropout_rate)(x)
#     outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 
#     ## Compiling model architecture
#     model.compile(
#         optimizer="adam", loss="binary_crossentropy",
#         metrics=["binary_accuracy"]
#     )
# 
#     return model
# 
# def tuner_fn(fn_args: FnArgs):
# 
#     ## Define splitted dataset transformed
#     tf_output = tft.TFTransformOutput(fn_args.transform_graph_path)
#     train_set = input_fn(fn_args.train_files[0], tf_output, num_epochs)
#     eval_set = input_fn(fn_args.eval_files[0], tf_output, num_epochs)
# 
#     ## Define vectorization layers
#     vectorizer_dataset = train_set.map(lambda f, _: f[transform_name(feature)])
#     vectorizer_layer = layers.TextVectorization(
#         max_tokens=1_000, output_mode="int", output_sequence_length=500
#     )
#     vectorizer_layer.adapt(vectorizer_dataset)
# 
#     ## Apply parameter tuning
#     tuner = kt.Hyperband(
#         hypermodel=lambda hp: model_builder(hp, vectorizer_layer),
#         objective=kt.Objective('binary_accuracy', direction='max'),
#         max_epochs=num_epochs,
#         factor=2,
#         directory=fn_args.working_dir,
#         project_name="kt_hyperband",
#     )
# 
#     return TunerResult(
#         tuner=tuner,
#         fit_kwargs={
#             "callbacks": [early_stopping],
#             "x": train_set,
#             "validation_data": eval_set,
#             "steps_per_epoch": fn_args.train_steps,
#             "validation_steps": fn_args.eval_steps,
#         },
#     )

# Define tuner component configuration
tuner = Tuner(
    module_file=os.path.abspath(TUNING_FILE),
    examples=transform.outputs["transformed_examples"],
    transform_graph=transform.outputs["transform_graph"],
    schema=schema_gen.outputs["schema"],
    train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=80),
    eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=20),
)

# Execute model params tuning on pipeline
INTERACTIVE_CONTEXT.run(tuner)

# NOTE: According keras_tuner hyperband documentation, trial iteration based on max_epochs * (math.log(max_epochs, factor) ** 2)
# Reference: https://github.com/keras-team/keras-tuner/issues/320

"""##### 4.2. Training Model"""

# Create script training file
TRAINER_FILE = 'train.py'

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRAINER_FILE}
# import os
# import tensorflow as tf
# import tensorflow_transform as tft
# import tensorflow_hub as hub
# from tensorflow.keras import layers
# from tfx.components.trainer.fn_args_utils import FnArgs
# 
# 
# FEATURE_KEY, LABEL_KEY = "message", "is_flight_intent"
# 
# 
# # ----------- Mengubah Nama Fitur
# def transformed_name(key):
#   """Renaming transformed features"""
#   return key + "_xf"
# 
# 
# # ----------- Memuat data dalam format TFRecord.
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
# 
# 
# # ----------- Memuat transformed_feature dan membagi dalam beberapa batch
# def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
#     transform_feature_spec = (
#         tf_transform_output.transformed_feature_spec().copy()
#     )
# 
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern,
#         batch_size=batch_size,
#         features=transform_feature_spec,
#         reader=gzip_reader_fn,
#         num_epochs=num_epochs,
#         label_key=transformed_name(LABEL_KEY),
#     )
#     return dataset
# 
# 
# # ---------- Arsitektur Model
# def model_builder(vectorizer_layer, hp):
#     inputs = tf.keras.Input(
#         shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
#     )
# 
#     x = vectorizer_layer(inputs)
#     x = layers.Embedding(input_dim=5000, output_dim=hp["embed_dims"])(x)
#     x = layers.Bidirectional(layers.LSTM(hp["lstm_units"]))(x)
# 
#     for _ in range(hp["num_hidden_layers"]):
#         x = layers.Dense(hp["dense_units"], activation=tf.nn.relu)(x)
#         x = layers.Dropout(hp["dropout_rate"])(x)
# 
#     outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
# 
#     model = tf.keras.Model(inputs=inputs, outputs = outputs)
# 
#     model.compile(
#         optimizer="adam",
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=[tf.keras.metrics.BinaryAccuracy()],
#     )
#     model.summary()
# 
#     return model
# 
# 
# # ---------- Menjalankan tahapan preprocessing data pada raw request data.
# def _get_serve_tf_examples_fn(model, tf_transform_output):
#     model.tft_layer = tf_transform_output.transform_features_layer()
# 
#     @tf.function
#     def serve_tf_examples_fn(serialized_tf_examples):
#         feature_spec = tf_transform_output.raw_feature_spec()
#         feature_spec.pop(LABEL_KEY)
#         parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
#         transformed_features = model.tft_layer(parsed_features)
# 
#         # get predictions using the transformed features
#         return model(transformed_features)
# 
#     return serve_tf_examples_fn
# 
# 
# # ----------- Run Training
# def run_fn(fn_args: FnArgs):
#     hp = fn_args.hyperparameters["values"]
#     log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
# 
#     # Define Tensorboard
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=log_dir, update_freq="batch")
# 
#     # Define Callback
#     early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy",
#                                                   min_delta=0,
#                                                   patience=12,
#                                                   verbose=0,
#                                                   mode="auto",
#                                                   baseline=None,
#                                                   restore_best_weights=True)
# 
#     # Define Model Check Poin Callbacks
#     model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir,
#                                                           monitor='val_binary_accuracy',
#                                                           mode='max',
#                                                           verbose=1,
#                                                           save_best_only=True)
# 
#     # Load the transform output
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
# 
#     # Create batches of data
#     train_set = input_fn(fn_args.train_files,
#                          tf_transform_output,
#                          hp['tuner/epochs'])
# 
#     eval_set = input_fn(fn_args.eval_files,
#                         tf_transform_output,
#                         hp['tuner/epochs'])
# 
#     vectorizer_dataset = train_set.map(
#         lambda f, l: f[transformed_name(FEATURE_KEY)])
# 
#     vectorizer_layer = layers.TextVectorization(
#         max_tokens=5000,
#         output_mode='int',
#         output_sequence_length=500
#     )
#     vectorizer_layer.adapt(vectorizer_dataset)
# 
#     # Build the model
#     model = model_builder(vectorizer_layer, hp)
# 
#     # Train the model
#     model.fit(x = train_set,
#               steps_per_epoch=fn_args.train_steps,
#               validation_data=eval_set,
#               validation_steps=fn_args.eval_steps,
#               callbacks=[tensorboard_callback,
#                          early_stop,
#                          model_checkpoint],
#               epochs=hp['tuner/epochs'],
#               verbose=1
#               )
# 
#     signatures = {
#         'serving_default': _get_serve_tf_examples_fn(model,
#                                                     tf_transform_output).get_concrete_function(tf.TensorSpec(shape=[None],
#                                                                                                              dtype=tf.string,
#                                                                                                              name='examples'))
#                   }
#     model.save(fn_args.serving_model_dir,
#                save_format='tf',
#                signatures=signatures)

# Define trainer component configuration
trainer = Trainer(
    module_file=os.path.abspath(TRAINER_FILE),
    examples = transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=80),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=20)
)

# Execute model training on pipeline
INTERACTIVE_CONTEXT.run(trainer)

"""#### 5. Model Validation

##### 5.1. Resolver Component to Overview Baseline Model Comparison
"""

# Create resolver configuration
model_resolver = Resolver(
    strategy_class = LatestBlessedModelStrategy,
    model = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')

# Execute resolver configuration
INTERACTIVE_CONTEXT.run(model_resolver)

"""##### 5.2. Evaluator Component to Evaluate Model using Tensorflow Model Analysis"""

# Define model evaluation metrics
eval_config = tf_ma.EvalConfig(
    model_specs=[tf_ma.ModelSpec(label_key='is_flight_intent')],
    slicing_specs=[tf_ma.SlicingSpec()],
    metrics_specs=[
        tf_ma.MetricsSpec(metrics=[
            tf_ma.MetricConfig(class_name='ExampleCount'),
            tf_ma.MetricConfig(class_name='AUC'),
            tf_ma.MetricConfig(class_name='FalsePositives'),
            tf_ma.MetricConfig(class_name='TruePositives'),
            tf_ma.MetricConfig(class_name='FalseNegatives'),
            tf_ma.MetricConfig(class_name='TrueNegatives'),
            tf_ma.MetricConfig(class_name='BinaryAccuracy',
                threshold=tf_ma.MetricThreshold(
                    value_threshold=tf_ma.GenericValueThreshold(
                        lower_bound={'value':0.5}),
                    change_threshold=tf_ma.GenericChangeThreshold(
                        direction=tf_ma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value':0.0001})
                    )
            )
        ])
    ]

)

# Define evaluator pipeline configuration based on defined metrics used
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)

# Execute evaluator pipeline
INTERACTIVE_CONTEXT.run(evaluator)

# Visualize evaluation result based on Tensorflow Model Analysis Libraries
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tf_ma.load_eval_result(eval_result)
tf_ma.view.render_slicing_metrics(tfma_result)
tf_ma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)

"""#### 6. Model Deployment

##### 6.1. Pusher Components
"""

# Define pusher configuration
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model_dir/intent-flight-classification'))
)

# Execute pusher component
INTERACTIVE_CONTEXT.run(pusher)

# Check file available
!ls

# Compress folder into .zip for submissions.
!zip -r pipeline-dicoding.zip /content -x "/content/__pycache__*" -x "/content/sample_data*"
