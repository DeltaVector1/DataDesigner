# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field

from .analysis.column_profilers import ColumnProfilerConfigT
from .base import ExportableConfigBase
from .columns import ColumnConfigT
from .models import ModelConfig
from .sampler_constraints import ColumnConstraintT
from .seed import SeedConfig


class DataDesignerConfig(ExportableConfigBase):
    """Configuration for NeMo Data Designer.

    This class defines the main configuration structure for NeMo Data Designer,
    which orchestrates the generation of synthetic data.

    Attributes:
        columns: Required list of column configurations defining how each column
            should be generated. Must contain at least one column.
        model_configs: Optional list of model configurations for LLM-based generation.
            Each model config defines the model, provider, and inference parameters.
        seed_config: Optional seed dataset settings to use for generation.
        constraints: Optional list of column constraints.
        profilers: Optional list of column profilers for analyzing generated data characteristics.
    """

    columns: list[ColumnConfigT] = Field(min_length=1)
    model_configs: list[ModelConfig] | None = None
    seed_config: SeedConfig | None = None
    constraints: list[ColumnConstraintT] | None = None
    profilers: list[ColumnProfilerConfigT] | None = None
