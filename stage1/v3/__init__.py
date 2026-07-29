"""Stage 1 v3: direct-observation label construction on Stage 0 v6 inputs."""

from stage1.v3.config import (
    FROZEN_REFERENCE_FIT_DATES,
    FROZEN_TEST_DATE,
    FROZEN_TRAIN_DATES,
    FROZEN_VALIDATION_DATES,
    STAGE1_V3_SCHEMA_VERSION,
    Stage1V3Config,
    Stage1V3ConfigError,
    load_config,
    validate_config,
    validate_split_config,
)
from stage1.v3.input_adapter import (
    BucketRef,
    Stage0Bucket,
    iter_stage0_buckets,
    load_stage0_bucket,
)
from stage1.v3.schema import ContractError, Stage1V3InputError

__all__ = [
    "BucketRef",
    "ContractError",
    "FROZEN_REFERENCE_FIT_DATES",
    "FROZEN_TEST_DATE",
    "FROZEN_TRAIN_DATES",
    "FROZEN_VALIDATION_DATES",
    "STAGE1_V3_SCHEMA_VERSION",
    "Stage0Bucket",
    "Stage1V3Config",
    "Stage1V3ConfigError",
    "Stage1V3InputError",
    "iter_stage0_buckets",
    "load_config",
    "load_stage0_bucket",
    "validate_config",
    "validate_split_config",
]
