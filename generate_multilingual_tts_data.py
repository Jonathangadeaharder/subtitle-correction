"""
Generates high-fidelity multilingual training and validation data.
Optimized version:
1. Generates TTS audio WAV files in parallel using concurrent ThreadPoolExecutor.
2. Transcribes the synthesized WAV files in batch loop using a single pre-loaded Whisper model.
"""
import json
import random
import subprocess
import sys
import concurrent.futures
from pathlib import Path
from mlx_audio.stt.generate import generate_transcription

LANG_CONFIGS = {
    "en": {
        "voices": ["Samantha", "Daniel", "Karen"],
        "templates": [
            "I want to {word1} my keys today.",
            "He went {word1} to find the answers.",
            "Did you hear {word1} call my name?",
            "She said {word1} was very happy.",
            "We should go {word1} tomorrow morning.",
        ],
        "confusions": [
            ("lose", "loose"), ("there", "their"), ("hear", "here"),
            ("your", "you're"), ("too", "to"), ("its", "it's")
        ]
    },
    "es": {
        "voices": ["Mónica", "Paulina"],
        "templates": [
            "Quiero {word1} este libro mañana.",
            "¿Has visto a {word1} hoy por la mañana?",
            "Espero que {word1} bien en tu viaje.",
            "No sé {word1} decir sobre esto.",
            "Ella va a {word1} una carta pronto.",
        ],
        "confusions": [
            ("ver", "haber"), ("valla", "vaya"), ("vaya", "valla"),
            ("hecho", "echo"), ("echo", "hecho"), ("casa", "caza"),
            ("caza", "casa"), ("halla", "haya"), ("haya", "halla")
        ]
    },
    "de": {
        "voices": ["Anna", "Eddy"],
        "templates": [
            "Ich möchte {word1} Buch lesen.",
            "Hast du {word1} heute gesehen?",
            "Es wäre {word1} wenn du kommst.",
            "Wir müssen {word1} nach Hause gehen.",
            "Er hat {word1} gesagt als er ging.",
        ],
        "confusions": [
            ("denn", "den"), ("den", "denn"), ("wäre", "wehre"),
            ("ihr", "sie"), ("sie", "ihr"), ("seite", "saite"),
            ("saite", "seite"), ("weg", "weck"), ("weck", "weg")
        ]
    },
    "fr": {
        "voices": ["Thomas", "Amélie"],
        "templates": [
            "Je veux {word1} ce livre aujourd'hui.",
            "As-tu vu {word1} ce matin ?",
            "Il est {word1} de faire cela.",
            "Nous allons {word1} à la maison.",
            "Elle a {word1} quelque chose de beau.",
        ],
        "confusions": [
            ("vert", "verre"), ("verre", "vert"), ("mer", "mère"),
            ("mère", "mer"), ("est", "es"), ("es", "est"),
            ("sain", "sein"), ("sein", "sain"), ("pain", "pin"),
            ("pin", "pain")
        ]
    }
}

def synthesize_single_tts(voice, text, wav_path):
    """Worker function to run say command in parallel."""
    try:
        subprocess.run([
            "say", "-v", voice,
            "-o", str(wav_path),
            "--data-format=LEI16@22050",
            text
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Synthesis failed: {e}", file=sys.stderr)
        return False

def generate_multilingual_batch(model_instance, items, start_idx):
    """
    Given a list of items: (lang, gt_sentence, voice, prefix)
    1. Synthesizes WAV files in parallel.
    2. Transcribes sequentially.
    """
    # 1. Parallel TTS Synthesis
    futures_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i, (lang, gt_sentence, voice, prefix) in enumerate(items):
            wav_path = Path(f"{prefix}_{start_idx + i}.wav")
            if wav_path.exists():
                wav_path.unlink()
            future = executor.submit(synthesize_single_tts, voice, gt_sentence, wav_path)
            futures_map[future] = (lang, gt_sentence, wav_path)
            
    # Wait for completion
    concurrent.futures.wait(futures_map.keys())
    
    # 2. Sequential Transcription
    results = []
    for future, (lang, gt_sentence, wav_path) in futures_map.items():
        if not future.result() or not wav_path.exists():
            continue
        try:
            segments = generate_transcription(
                model=model_instance,
                audio=str(wav_path),
                language=lang
            )
            transcription = getattr(segments, "text", "").strip()
            if transcription:
                results.append({
                    "whisper_text": transcription,
                    "corrected_text": gt_sentence,
                    "ground_truth": gt_sentence,
                    "lang": lang,
                    "source_file": "synthetic_tts"
                })
        except Exception as e:
            print(f"Transcription failed: {e}", file=sys.stderr)
        finally:
            if wav_path.exists():
                wav_path.unlink()
                
    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: generate_multilingual_tts_data.py <train_count> <val_count>")
        sys.exit(1)
        
    train_count = int(sys.argv[1])
    val_count = int(sys.argv[2])
    
    print("Loading Whisper model into memory...")
    from mlx_audio.stt.utils import load_model
    model_instance = load_model("mlx-community/whisper-large-v3-turbo")
    
    langs = ["en", "es", "de", "fr"]
    train_per_lang = train_count // len(langs)
    
    # 1. Generate and append training data
    subcache_path = Path.home() / "Downloads" / ".subcache" / "training_pairs.jsonl"
    generated_train = 0
    seed = 42
    
    for lang in langs:
        print(f"Preparing batch templates for '{lang}'...")
        rng = random.Random(seed)
        cfg = LANG_CONFIGS[lang]
        
        # Prepare list of metadata for parallel generation
        items = []
        for _ in range(train_per_lang):
            conf = rng.choice(cfg["confusions"])
            w_word, gt_word = conf
            template = rng.choice(cfg["templates"])
            gt_sentence = template.format(word1=gt_word).strip()
            voice = rng.choice(cfg["voices"])
            items.append((lang, gt_sentence, voice, "train"))
            
        print(f"Processing {train_per_lang} training examples in parallel batches...")
        batch_size = 50
        for offset in range(0, len(items), batch_size):
            batch = items[offset:offset+batch_size]
            pairs = generate_multilingual_batch(model_instance, batch, offset)
            with open(subcache_path, "a") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            generated_train += len(pairs)
            print(f"  Processed {offset + len(batch)}/{train_per_lang}...")
            
        seed += 1
        
    print(f"Successfully generated {generated_train} training examples.")

    # 2. Generate and append validation data
    eval_dataset_path = Path("evaluation/dataset.jsonl")
    val_per_lang = val_count // len(langs)
    generated_val = 0
    
    if eval_dataset_path.exists():
        with open(eval_dataset_path) as f:
            existing = [json.loads(line) for line in f if line.strip()]
    else:
        existing = []
        
    start_id = len(existing) + 1
    
    for lang in langs:
        print(f"Preparing batch templates for validation '{lang}'...")
        rng = random.Random(seed)
        cfg = LANG_CONFIGS[lang]
        
        items = []
        for _ in range(val_per_lang):
            conf = rng.choice(cfg["confusions"])
            w_word, gt_word = conf
            template = rng.choice(cfg["templates"])
            gt_sentence = template.format(word1=gt_word).strip()
            voice = rng.choice(cfg["voices"])
            items.append((lang, gt_sentence, voice, "val"))
            
        print(f"Processing {val_per_lang} validation examples in parallel batches...")
        batch_size = 20
        for offset in range(0, len(items), batch_size):
            batch = items[offset:offset+batch_size]
            pairs = generate_multilingual_batch(model_instance, batch, offset)
            for pair in pairs:
                existing.append({
                    "id": f"slice_tts_{start_id + generated_val:04d}",
                    "audio_path": "",
                    "whisper_text": pair["whisper_text"],
                    "ground_truth": pair["ground_truth"],
                    "reference": pair["corrected_text"],
                    "source_file": "active_learning_val",
                    "start_time": "00:00:00,000",
                    "end_time": "00:00:00,000",
                    "has_error": pair["whisper_text"] != pair["ground_truth"]
                })
                generated_val += 1
            print(f"  Processed {offset + len(batch)}/{val_per_lang}...")
            
        seed += 1
        
    with open(eval_dataset_path, "w") as f:
        for entry in existing:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {generated_val} validation examples.")

if __name__ == "__main__":
    main()
