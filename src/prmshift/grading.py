"""Conservative answer grading. Unparsed or unsupported answers remain unknown.

This is not a general proof checker. Numeric arithmetic uses an allowlisted AST,
never eval/sympify on untrusted model output. Supported numeric inequalities are
not silently interpreted as final answers.
"""
from __future__ import annotations
import ast
from fractions import Fraction
import re


def extract_answer(text):
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for m in reversed(starts):
        depth, start = 1, m.end()
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start:i].strip()
    if "####" in text:
        return text.rsplit("####", 1)[1].strip().splitlines()[0]
    matches = list(re.finditer(r"(?im)^\s*(?:final answer|answer)\s*[:=]\s*(.+)$", text))
    return matches[-1].group(1).strip() if matches else None


def normalize(text):
    text = str(text).strip().strip("$").strip()
    text = text.replace(r"\,", "").replace(r"\!", "").replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\s+", "", text)


def numeric(text):
    text = normalize(text)
    if len(text) > 200:
        raise ValueError("expression too long")
    for _ in range(10):
        updated = re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.replace("^", "**").replace(r"\times", "*").replace(r"\cdot", "*")
    if not re.fullmatch(r"[0-9.eE+*/()\- ]+", text):
        raise ValueError("unsupported expression")
    if any(abs(int(exponent)) > 1000 for exponent in re.findall(r"[eE]([+-]?\d+)", text)):
        raise ValueError("oversized exponent")
    tree = ast.parse(text, mode="eval")

    def bounded(value):
        if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
            raise ValueError("oversized intermediate")
        return value

    def walk(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            # Preserve the original decimal literal; AST float conversion would
            # conflate distinct long decimals through IEEE rounding.
            return bounded(Fraction(ast.get_source_segment(text, n)))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = walk(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            a, b = walk(n.left), walk(n.right)
            if isinstance(n.op, ast.Add): return bounded(a + b)
            if isinstance(n.op, ast.Sub): return bounded(a - b)
            if isinstance(n.op, ast.Mult): return bounded(a * b)
            if isinstance(n.op, ast.Div): return bounded(a / b)
            if isinstance(n.op, ast.Pow) and b.denominator == 1 and abs(b) <= 12: return bounded(a ** int(b))
        raise ValueError("unsupported AST")

    value = walk(tree.body)
    if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
        raise ValueError("oversized result")
    return value


def grade_answer(predicted, reference):
    if predicted is None or reference is None or not str(predicted).strip() or not str(reference).strip():
        return {"label": None, "reason": "missing_answer"}
    if normalize(predicted) == normalize(reference):
        return {"label": True, "reason": "normalized_exact_match"}
    try:
        a, b = numeric(predicted), numeric(reference)
        return {"label": bool(a == b), "reason": "exact_numeric_comparison"}
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return {"label": None, "reason": "unsupported_equivalence_requires_review"}
