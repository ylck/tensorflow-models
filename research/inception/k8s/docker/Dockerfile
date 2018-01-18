FROM gcr.io/tensorflow/tensorflow:latest-devel

ADD inception /training

WORKDIR /training
RUN bazel build //inception:download_and_preprocess_imagenet
RUN bazel build //inception:download_and_preprocess_flowers
RUN bazel build //inception:imagenet_train
RUN bazel build //inception:imagenet_distributed_train
RUN bazel build //inception:imagenet_eval
