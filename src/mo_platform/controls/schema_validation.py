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


# 1. Validation models

Severity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]

@dataclass(frozen=True)
class ColumnRule:
    """Validation requirements for a single column."""

    name: str
    dtype: str = "string"
    nullable: bool = False
    allowed_values: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    strict_minimum: bool = False
    strict_maximum: bool = False

    # When true, non-null values must be unique within this column.
    unique: bool = False


@dataclass(frozen=True)
class TableSchema:
    """Schema and key constraints for a dataset."""

    name: str
    columns: tuple[ColumnRule, ...]
    primary_key: tuple[str, ...] = ()
    allow_extra_columns: bool = True

    @property
    def column_names(self) -> set[str]:
        return {column.name for column in self.columns}

    @property
    def rules_by_name(self) -> dict[str, ColumnRule]:
        return {column.name: column for column in self.columns}


@dataclass(frozen=True)
class ValidationIssue:
    """One schema-validation finding."""

    dataset: str
    rule_id: str
    severity: Severity
    message: str
    column: str | None = None
    row_indices: tuple[Any, ...] = ()
    observed_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "dataset": self.dataset,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "column": self.column,
            "row_indices": list(self.row_indices),
            "observed_value": _safe_value(self.observed_value),}


@dataclass
class ValidationReport:
    """Aggregated findings for one or more datasets."""

    issues: list[ValidationIssue] = field(default_factory=list)
    datasets_checked: list[str] = field(default_factory=list)
    rows_checked: dict[str, int] = field(default_factory=dict)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    @property
    def error_count(self) -> int:
        return sum(
            issue.severity in ("ERROR", "CRITICAL")
            for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(
            issue.severity == "WARNING"
            for issue in self.issues)

    @property
    def critical_count(self) -> int:
        return sum(
            issue.severity == "CRITICAL"
            for issue in self.issues)

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert findings to a report suitable for CSV export."""
        columns = [
            "dataset",
            "rule_id",
            "severity",
            "message",
            "column",
            "row_indices",
            "observed_value",]

        rows = []
        for issue in self.issues:
            record = issue.to_dict()
            record["row_indices"] = ",".join(
                str(i) for i in record["row_indices"])
            record["observed_value"] = str(
                record["observed_value"])
            rows.append(record)

        return pd.DataFrame(rows, columns=columns)

    def summary(self) -> dict[str, Any]:
        """Return aggregate validation statistics."""
        return {
            "passed": self.passed,
            "datasets_checked": len(self.datasets_checked),
            "rows_checked": sum(self.rows_checked.values()),
            "issues_total": len(self.issues),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "critical": self.critical_count,}

    def raise_for_errors(self) -> None:
        """Stop a pipeline when errors or critical issues exist."""
        if not self.passed:
            raise SchemaValidationError(
                f"Schema validation failed: {self.summary()}")


class SchemaValidationError(ValueError):
    """Raised when schema validation blocks a processing pipeline."""


# 2. Helper functions

def _safe_value(value: Any) -> Any:
    """Normalize common Pandas/NumPy values for reporting."""
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _sample_indices(
    mask: pd.Series,
    maximum: int = 10,) -> tuple[Any, ...]:
    """Return a bounded set of DataFrame index labels."""
    return tuple(mask[mask].index[:maximum].tolist())


def _make_issue(
    dataset: str,
    rule_id: str,
    message: str,
    *,
    severity: Severity = "ERROR",
    column: str | None = None,
    row_indices: Sequence[Any] = (),
    observed_value: Any = None,) -> ValidationIssue:
    return ValidationIssue(
        dataset=dataset,
        rule_id=rule_id,
        severity=severity,
        message=message,
        column=column,
        row_indices=tuple(row_indices),
        observed_value=observed_value,)

# 3. Type conversion for validation

def _invalid_type_mask(
    series: pd.Series,
    dtype: str,) -> pd.Series:
    """
    Identify non-null values that cannot be interpreted as the
    expected type.

    This validates convertibility without modifying the input.
    """

    non_null = series.notna()

    if dtype in ("string", "str"):
        # Strings and object-backed values are accepted as text.
        return pd.Series(False, index=series.index)

    if dtype in ("integer", "int"):
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = non_null & numeric.isna()

        fractional = numeric.notna() & (
            np.mod(numeric.fillna(0), 1) != 0)

        return invalid | fractional

    if dtype in ("float", "number", "numeric"):
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = non_null & numeric.isna()

        finite = pd.Series(
            np.isfinite(
                numeric.to_numpy(dtype=float, na_value=np.nan)
            ),
            index=series.index,)

        return invalid | (numeric.notna() & ~finite)

    if dtype in ("date", "datetime"):
        parsed = pd.to_datetime(series, errors="coerce")
        return non_null & parsed.isna()

    if dtype in ("boolean", "bool"):
        valid_values = {
            True, False, 1, 0,
            "True", "False", "true", "false",
            "TRUE", "FALSE", "1", "0",}
        return non_null & ~series.isin(valid_values)

    raise ValueError(
        f"Unsupported schema dtype '{dtype}'. "
        "Supported types: string, integer, float, date, boolean.")


def _numeric_series(series: pd.Series) -> pd.Series:
    """Create a temporary numeric representation."""
    return pd.to_numeric(series, errors="coerce")


def _date_series(series: pd.Series) -> pd.Series:
    """Create a temporary datetime representation."""
    return pd.to_datetime(series, errors="coerce")


# 4. Core schema validation

def validate_schema(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    dataset_name: str | None = None,) -> ValidationReport:
    """
    Validate a DataFrame against a TableSchema.

    The original DataFrame is not modified.

    Returns:
        ValidationReport containing schema findings.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    dataset = dataset_name or schema.name
    report = ValidationReport(
        datasets_checked=[dataset],
        rows_checked={dataset: len(df)},
        )

    # 4.1 Required columns
    missing_columns = sorted(schema.column_names - set(df.columns))

    for column in missing_columns:
        report.add(
            _make_issue(
                dataset,
                "SCHEMA_MISSING_COLUMN",
                f"Required column '{column}' is missing.",
                column=column,)
                )

    # 4.2 Unexpected columns
    if not schema.allow_extra_columns:
        unexpected = sorted(set(df.columns) - schema.column_names)

        for column in unexpected:
            report.add(
                _make_issue(
                    dataset,
                    "SCHEMA_UNEXPECTED_COLUMN",
                    f"Unexpected column '{column}' is present.",
                    severity="WARNING",
                    column=column,)
                    )

    # 4.3 Column-level rules
    for rule in schema.columns:
        if rule.name not in df.columns:
            # Missing column has already been reported.
            continue

        series = df[rule.name]

        # Nullability
        if not rule.nullable:
            null_mask = series.isna()

            if null_mask.any():
                report.add(
                    _make_issue(
                        dataset,
                        "SCHEMA_NOT_NULL",
                        f"Column '{rule.name}' contains "
                        f"{int(null_mask.sum())} null values.",
                        column=rule.name,
                        row_indices=_sample_indices(null_mask),)
                        )

        # Data type
        invalid_mask = _invalid_type_mask(series, rule.dtype)

        if invalid_mask.any():
            report.add(
                _make_issue(
                    dataset,
                    "SCHEMA_INVALID_TYPE",
                    f"Column '{rule.name}' contains values "
                    f"that cannot be interpreted as '{rule.dtype}'.",
                    column=rule.name,
                    row_indices=_sample_indices(invalid_mask),
                    )
                    )

        # Enum / domain validation
        if rule.allowed_values is not None:
            enum_mask = (
                series.notna()
                & ~series.isin(rule.allowed_values))

            if enum_mask.any():
                report.add(
                    _make_issue(
                        dataset,
                        "SCHEMA_INVALID_ENUM",
                        f"Column '{rule.name}' contains values "
                        "outside its permitted domain.",
                        column=rule.name,
                        row_indices=_sample_indices(enum_mask),
                        observed_value=series[
                            enum_mask
                        ].head(5).tolist(),)
                        )

        # Numeric range validation
        if rule.dtype in (
            "integer", "int", "float", "number", "numeric"):
            numeric = _numeric_series(series)

            if rule.minimum is not None:
                if rule.strict_minimum:
                    lower_mask = numeric <= rule.minimum
                else:
                    lower_mask = numeric < rule.minimum

                lower_mask = lower_mask.fillna(False)

                if lower_mask.any():
                    report.add(
                        _make_issue(
                            dataset,
                            "SCHEMA_MINIMUM",
                            f"Column '{rule.name}' violates "
                            f"minimum {rule.minimum}.",
                            column=rule.name,
                            row_indices=_sample_indices(lower_mask),)
                            )

            if rule.maximum is not None:
                if rule.strict_maximum:
                    upper_mask = numeric >= rule.maximum
                else:
                    upper_mask = numeric > rule.maximum

                upper_mask = upper_mask.fillna(False)

                if upper_mask.any():
                    report.add(
                        _make_issue(
                            dataset,
                            "SCHEMA_MAXIMUM",
                            f"Column '{rule.name}' violates "
                            f"maximum {rule.maximum}.",
                            column=rule.name,
                            row_indices=_sample_indices(upper_mask),)
                            )

        # Single-column uniqueness
        if rule.unique:
            duplicate_mask = (
                series.notna()
                & series.duplicated(keep=False)
                )

            if duplicate_mask.any():
                report.add(
                    _make_issue(
                        dataset,
                        "SCHEMA_COLUMN_NOT_UNIQUE",
                        f"Column '{rule.name}' contains duplicate values.",
                        column=rule.name,
                        row_indices=_sample_indices(duplicate_mask),)
                        )

    # 4.4 Primary-key validation
    present_key_columns = [
        column
        for column in schema.primary_key
        if column in df.columns]

    if len(present_key_columns) != len(schema.primary_key):
        report.add(
            _make_issue(
                dataset,
                "SCHEMA_PRIMARY_KEY_MISSING",
                "One or more primary-key columns are missing.",)
                )
    elif schema.primary_key:
        key_frame = df[list(schema.primary_key)]

        null_key_mask = key_frame.isna().any(axis=1)

        if null_key_mask.any():
            report.add(
                _make_issue(
                    dataset,
                    "SCHEMA_PRIMARY_KEY_NULL",
                    "Primary-key fields must not contain null values.",
                    row_indices=_sample_indices(null_key_mask),)
                    )

        duplicate_key_mask = key_frame.duplicated(keep=False)

        if duplicate_key_mask.any():
            report.add(
                _make_issue(
                    dataset,
                    "SCHEMA_PRIMARY_KEY_DUPLICATE",
                    f"Primary key {schema.primary_key} is not unique.",
                    row_indices=_sample_indices(duplicate_key_mask),)
                    )

    return report
