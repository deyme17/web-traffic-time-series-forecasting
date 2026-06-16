from .lstm_baseline import BaselineLSTM
from .lstm_conv_attn import ConvAttnLSTM
from .gru_conv_attn import ConvAttnGRU
from utils import Registry

MODELS = Registry()

MODELS.register("LSTM")(BaselineLSTM)
MODELS.register("LSTM_ConvAttn")(ConvAttnLSTM)
MODELS.register("GRU_ConvAttn")(ConvAttnLSTM)