import dataiku
import pandas as pd
import sys
sys.path.append('python-lib/')
import os
import random
from datetime import datetime

from dataiku.customrecipe import *
import logging
import dku_timeseries

from dku_timeseries import ResamplerParams, Resampler
from dku_tools import get_resampling_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')

import dataikuapi

host = "http://localhost:11200"
apiKey = "k2pPY8TnIg7HqDHGazoMiKYaNZVgh307"
client = dataikuapi.DSSClient(host, apiKey)

client.list_project_keys()

dataiku.set_remote_dss("http://localhost:11200", "k2pPY8TnIg7HqDHGazoMiKYaNZVgh307")


dataset = dataiku.Dataset("DATACIENCENETMAIF.stocks_prepared")

df = dataset.get_dataframe()
