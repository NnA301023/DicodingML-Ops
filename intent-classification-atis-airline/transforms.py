import tensorflow as tf
feature, label = 'message', 'is_flight_intent'

def transform_name(key: str) -> str:
    return f"{key}_xf"

def preprocessing_fn(inputs):
    outputs = {
        transform_name(feature): tf.strings.lower(inputs[feature]),
        transform_name(label): tf.cast(inputs[label], tf.int64)
    }
    return outputs
