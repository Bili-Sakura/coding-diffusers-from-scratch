# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

######################################################################
# Copyright 2025 Sakura. All rights reserved.
# The following code is largely inherited from `huggingface/diffuser` but we provide a much more simplified version.

# Sakura: For simplicity, we re-use built-in functions and classes from `diffusers` which we will re-implement later on.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from utils import *
