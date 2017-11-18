NNDDoSForecast
=====
NNDDosForecast is a stream based hierarchical LSTM model for DDoS attack forecasting. 

The input is a tweet stream, and output is the attack probability of the target in a certain day.

The codes are the implementation of following paper:

Zhongqing Wang, and Yue Zhang. DDoS Event Forecasting using Twitter Data. In Proceeding of IJCAI-17.

http://static.ijcai.org/proceedings-2017/0580.pdf

Prerequisition
=====
Keras 2.0.6

Usage
=====
python main.py

Following are four kinds of stream models which are described in the paper.

* CNN based Vanilla Stream Model:
cnn_prediction()

* LSTM based Vanilla Stream Model:
lstm_prediction()

* Short- and Long-Term Stream Model:
lstmMulti()

* Hierarchical Stream Model:
lstmMulti2()