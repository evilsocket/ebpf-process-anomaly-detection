Process activity anomaly detection using eBPF system call tracing and unsupervised learning Autoencoders.

## Installation

```sh
sudo pip3 install -r requirements.txt
```

## Learning

During the first step, we need to sample as much data as possible from a running target process (pid 1234 in this example):

```sh
sudo ./main.py --pid 1234 --data activity.csv --learn
```

Keep the sampling going while you trigger normal behaviour in the target process, this will generate the `activity.csv` file for training.

## Training a model

We'll now train a model to detect anomalies:

```sh
./main.py --train --data activity.csv --model model.h5
```

The autoencoder saved to `model.h5` can now be used for anomaly detection with the error threshold print at the end of the training.

## Anomaly detection

Once the model has been trained it can be used on the target process to detect anomalies, in this case we're using a 10.0 error threshold:

```sh
sudo ./main.py --pid 1234 --model model.h5 --max-error 10.0 --run
```

When an anomaly is detected the cumulative error will be printed along wiht the top 3 anomalous system calls:

```
error = 30.605255 - max = 10.000000 - top 3:
  b'getpriority' = 0.994272
  b'writev' = 0.987554
  b'creat' = 0.969955
```

## License

This project is made with ♥  by [@evilsocket](https://twitter.com/evilsocket) and it is released under the GPL3 license.