FROM tensorflow/serving:latest

COPY ./serving_model_dir/intent-flight-classification /models/intent-flight-classification

ENV MODEL_NAME=intent-flight-classification

CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=intent-flight-classification", "--model_base_path=/models/intent-flight-classification"]