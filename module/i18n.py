from .settings import load_value

LANGUAGES = [
    ("ja", "日本語"),
    ("en", "English"),
]

_current = None


def _detect_language():
    lang = str(load_value("language", "ja") or "ja").strip().lower()

    if any(code == lang for code, _ in LANGUAGES):
        return lang

    return "ja"


def get_language():
    global _current

    if _current is None:
        _current = _detect_language()

    return _current


def set_language(lang):
    global _current
    _current = lang if any(code == lang for code, _ in LANGUAGES) else "ja"


def tr(text, en_text=None):
    if get_language() == "en" and en_text is not None:
        return en_text

    return text
