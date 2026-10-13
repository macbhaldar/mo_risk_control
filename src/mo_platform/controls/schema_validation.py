"""Schema validation framework for the Middle Office Risk Management
platform.

Responsibilities
1. Required-column and unexpected-column checks.
2. Data-type validation.
3. Nullability and enumerated-value checks.
4. Numeric range validation.
5. Primary-key uniqueness.
6. Date validation and date-order checks.
7. Foreign-key / referential-integrity validation.
8. Structured validation reports suitable for audit trails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

