"""Helsinki-NLP Marian models used by multilingual synthetic data generation."""

TRANSLATION_MODELS = {
    "en": {
        "forward": "Helsinki-NLP/opus-mt-en-es",
        "backward": "Helsinki-NLP/opus-mt-es-en",
    },
    "es": {
        "forward": "Helsinki-NLP/opus-mt-es-en",
        "backward": "Helsinki-NLP/opus-mt-en-es",
    },
    "de": {
        "forward": "Helsinki-NLP/opus-mt-de-en",
        "backward": "Helsinki-NLP/opus-mt-en-de",
    },
    "fr": {
        "forward": "Helsinki-NLP/opus-mt-fr-en",
        "backward": "Helsinki-NLP/opus-mt-en-fr",
    },
}
