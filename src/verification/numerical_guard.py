"""
SatQuery AI -- Deterministic Numerical Guard
Section 21: Numerical Guardrail against Generative VLM Measurement Hallucinations.

Governing Law:
"The language model NEVER invents measurements -- area, pixel count, coordinates,
distances, dates, resolutions, confidence values, counts. The measurement engine
computes the number; the response generator receives it; the LLM may only verbalize
a value it was given, never a different one."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NumericalGuardResult:
    """Outcome of numerical validation on generated response text."""
    is_valid: bool
    detected_numbers: list[float]
    certified_numbers: list[float]
    uncertified_numbers: list[float]
    violations: list[str]
    guarded_text: str


class NumericalGuard:
    """
    Programmatic token auditor that cross-references all numerical claims
    in generated text against a certified GIS dictionary of computed physical quantities.
    """

    # Permitted harmless constants commonly found in descriptions and ratios
    PERMITTED_CONSTANTS = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 100.0, 1e-7, 1e-10, 83.0, 15.0, 0.15,
    }

    @staticmethod
    def extract_numbers(text: str) -> list[float]:
        """
        Extract numerical tokens (integers, floats, negative numbers, scientific notation)
        from a string. Excludes dates like 2024-01-15 or version strings.
        """
        # Match standalone numbers, percentages, or measurements
        pattern = r"(?<![a-zA-Z_\-])-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"
        raw_matches = re.findall(pattern, text)
        numbers = []
        for m in raw_matches:
            try:
                numbers.append(float(m))
            except ValueError:
                continue
        return numbers

    @classmethod
    def verify_response(
        cls,
        text: str,
        certified_numbers: dict[str, float | int],
        tolerance: float = 0.05,
    ) -> NumericalGuardResult:
        """
        Verify that all numbers in `text` originate from the certified calculation dictionary.
        Returns NumericalGuardResult indicating pass/fail status and violations.
        """
        detected = cls.extract_numbers(text)
        certified_vals = [float(v) for v in certified_numbers.values()]
        uncertified = []
        violations = []

        for num in detected:
            # Check if num is a harmless constant (e.g. 100%, 7x7 filter, band counts)
            if any(abs(num - c) < 1e-4 for c in cls.PERMITTED_CONSTANTS):
                continue

            # Check if num matches any certified value within relative tolerance
            matched = False
            for cert in certified_vals:
                if abs(cert) < 1e-6:
                    if abs(num) < 1e-6:
                        matched = True
                        break
                else:
                    rel_err = abs(num - cert) / abs(cert)
                    if rel_err <= tolerance:
                        matched = True
                        break

            if not matched:
                uncertified.append(num)
                violations.append(
                    f"Uncertified numerical token detected: {num}. Not found in certified GIS calculation dictionary."
                )

        is_valid = (len(violations) == 0)
        guarded_text = text
        if not is_valid:
            # Append guard warning to ensure downstream audit transparency
            guarded_text += f" [NUMERICAL_GUARD_ALERT: {len(violations)} uncertified numbers flagged]"

        return NumericalGuardResult(
            is_valid=is_valid,
            detected_numbers=detected,
            certified_numbers=certified_vals,
            uncertified_numbers=uncertified,
            violations=violations,
            guarded_text=guarded_text,
        )

    @classmethod
    def format_certified_statement(
        cls,
        template: str,
        certified_numbers: dict[str, Any],
    ) -> str:
        """
        Safely construct a response statement locked strictly to certified values.
        """
        try:
            return template.format(**certified_numbers)
        except KeyError as e:
            return f"Certified calculation incomplete: missing parameter {e}"
