
from typing import Optional
import argparse

import torch
import random
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast
from transformers import XLNetTokenizer, XLNetModel
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import os
import json
import time
from tqdm import tqdm
from random import shuffle
from datetime import date, datetime

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.integration.wandb import WeightsAndBiasesCallback

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)