"""GSM8K numeric extraction with exact rational comparison (no float rounding)."""
import re
from fractions import Fraction

NUMBER = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+)?'


def extract_answer(text, gold=False):
    # Explicit final markers take priority; no fallback to an earlier calculation
    # when a final marker is present but malformed.
    mode = 'last_number'
    if '####' in text:
        text = text.rsplit('####', 1)[1]
        mode = 'marker'
    elif gold:
        return None, 'failure'
    else:
        boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
        if boxed:
            text, mode = boxed[-1], 'boxed'
        else:
            markers = list(re.finditer(r'(?:final answer|answer is|answer:)\s*[:=]?\s*', text, re.I))
            if markers:
                text, mode = text[markers[-1].end():], 'answer_phrase'
    matches = list(re.finditer(NUMBER, text))
    if not matches:
        return None, 'failure'
    value = matches[-1].group() if mode == 'last_number' else matches[0].group()
    try:
        value = value.replace(',', '').replace(' ', '')
        if '/' in value:
            a, b = value.split('/')
            number = Fraction(a) / Fraction(b)
        else:
            number = Fraction(value)
        return str(number), mode
    except (ValueError, ZeroDivisionError):
        return None, 'failure'
