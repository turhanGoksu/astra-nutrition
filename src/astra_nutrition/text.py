"""Text normalization shared by amount parsing and food matching."""

_TURKISH_FOLD = str.maketrans(
    {
        "İ": "i",
        "I": "i",
        "ı": "i",
        "Ş": "s",
        "ş": "s",
        "Ğ": "g",
        "ğ": "g",
        "Ü": "u",
        "ü": "u",
        "Ö": "o",
        "ö": "o",
        "Ç": "c",
        "ç": "c",
        "Â": "a",
        "â": "a",
        "Î": "i",
        "î": "i",
        "Û": "u",
        "û": "u",
    }
)


# Words that say when or how a meal was eaten, not what was eaten. The parser
# sometimes glues them to a food name ("Akşam Biber Dolması") or uses them as
# the amount ("Menemen [sabah]"). Folded (see fold).
MEAL_TIME_WORDS = frozenset(
    {
        "sabah", "sabahleyin", "ogle", "oglen", "ogleden", "aksam", "aksamleyin",
        "gece", "kahvalti", "kahvaltida", "kahvaltiya", "ara", "ogun", "yemegi",
        "yemeginde", "breakfast", "lunch", "dinner", "brunch", "snack",
    }
)  # fmt: skip
EATING_VERBS = frozenset(
    {
        "yedim", "yendim", "yedik", "ictim", "ictik", "yaptim", "yaptik", "aldim",
        "atistirdim", "bandim",
    }
)  # fmt: skip


def fold(text: str) -> str:
    """Lowercase, fold Turkish letters to ASCII and collapse whitespace.

    ``str.lower()`` is not Turkish-aware: ``"İ".lower()`` is "i" plus a
    combining dot, so ``"İki".lower() != "iki"``. Folding both user text and
    our own vocabularies to plain ASCII avoids that trap and also matches
    users who type without Turkish characters ("tavuk gogsu", "kasik").
    """
    return " ".join(text.translate(_TURKISH_FOLD).lower().split())


def embedding_text(text: str, lowercase: bool = True) -> str:
    """Normalize text before embedding: lowercase, but keep Turkish letters.

    The embedding model is case-sensitive for Turkish ("tavuk göğsü" vs
    "Tavuk göğsü": 0.82 similarity), so aliases and queries must share casing.
    ASCII folding is avoided here because it also lowered similarity (0.71).
    Known limit: a capital "I" becomes "i", although in Turkish words it
    stands for "ı" ("Izgara" -> "izgara", not "ızgara").
    """
    if not lowercase:
        return " ".join(text.split())
    return " ".join(text.replace("İ", "i").lower().split())
