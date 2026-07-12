from subtitle_correction.translation_models import TRANSLATION_MODELS


def test_translation_models_cover_supported_synthetic_languages() -> None:
    assert set(TRANSLATION_MODELS) == {"en", "es", "de", "fr"}
    for lang, models in TRANSLATION_MODELS.items():
        assert "forward" in models and "backward" in models
        assert models["forward"].startswith("Helsinki-NLP/")
        assert models["backward"].startswith("Helsinki-NLP/")
