from .lstm_baseline import BaselineLSTM
from .lstm_conv_attn import ConvAttnLSTM
from utils import Registry

MODELS = Registry()

MODELS.register("LSTM")(BaselineLSTM)
MODELS.register("LSTM_ConvAttn")(ConvAttnLSTM)