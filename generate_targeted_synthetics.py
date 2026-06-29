"""
Generates targeted synthetic training pairs to fix specific gaps:
1. Preservation of line breaks (\\n) from Reference.
2. Preservation of music notes (♪) from Reference.
3. Plurality/grammar agreement with Reference (e.g. uncle -> uncles).
Appends these pairs to /Users/jonathangadeaharder/Downloads/.subcache/training_pairs.jsonl.
"""
import json
from pathlib import Path

TARGETED_PAIRS = [
    # --- Group 1: Newline Preservation ---
    {
        "whisper_text": "I know how painful it is to lose.",
        "corrected_text": "I know how painful it is\nto lose Lynette,",
        "ground_truth": "I know how painful it is\nto lose Lynette,",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "I've never seen anyone so beautiful.",
        "corrected_text": "I've never seen anyone\nlook so beautiful.",
        "ground_truth": "I've never seen anyone\nlook so beautiful.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "We want to build a snowman?",
        "corrected_text": "You want\nto build a snowman?",
        "ground_truth": "You want\nto build a snowman?",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "I think it's in one of the buildings.",
        "corrected_text": "I think it's in one\nof the buildings here.",
        "ground_truth": "I think it's in one\nof the buildings here.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "She was the kind of person you want to hold.",
        "corrected_text": "She was the kind of person\nyou want to hang on to.",
        "ground_truth": "She was the kind of person\nyou want to hang on to.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "I have a friend who needs peace.",
        "corrected_text": "I have a friend\nwho needs a piece.",
        "ground_truth": "I have a friend\nwho needs a piece.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "I am standing before you and I see you.",
        "corrected_text": "I'm standing here\nlooking at you.",
        "ground_truth": "I'm standing here\nlooking at you.",
        "source_file": "synthetic_targeted"
    },
    
    # --- Group 2: Music Notes (♪) Preservation ---
    {
        "whisper_text": "And I say to myself",
        "corrected_text": "♪ and I say to myself ♪",
        "ground_truth": "♪ and I say to myself ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "It's wonderful, wonderful",
        "corrected_text": "♪ it's wonderful, wonderful ♪",
        "ground_truth": "♪ it's wonderful, wonderful ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "Oh, so wonderful, my love",
        "corrected_text": "♪ oh, so wonderful, my love ♪",
        "ground_truth": "♪ oh, so wonderful, my love ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "The bloodhounds are hot on the trail",
        "corrected_text": "♪ the bloodhounds\nare hot on my trail ♪",
        "ground_truth": "♪ the bloodhounds\nare hot on my trail ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "Last evening I shot down my sweetheart",
        "corrected_text": "♪ last evening,\nI shot down my sweetheart ♪",
        "ground_truth": "♪ last evening,\nI shot down my sweetheart ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "This morning I broke out of jail",
        "corrected_text": "♪ this morning,\nI broke out of jail ♪",
        "ground_truth": "♪ this morning,\nI broke out of jail ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "For now we're ahead of the posse",
        "corrected_text": "♪ one hour ahead of the posse ♪",
        "ground_truth": "♪ one hour ahead of the posse ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "I'm nothing in between.",
        "corrected_text": "♪ I'm nothin' in between",
        "ground_truth": "♪ I'm nothin' in between ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "Colors the tiles and the bathroom mat",
        "corrected_text": "♪ Covers the tiles and the bathroom mat",
        "ground_truth": "♪ Covers the tiles and the bathroom mat ♪",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "How they fail doesn't matter",
        "corrected_text": "♪ How they feel doesn't matter",
        "ground_truth": "♪ How they feel doesn't matter ♪",
        "source_file": "synthetic_targeted"
    },
    
    # --- Group 3: Plurality & Grammar Agreement ---
    {
        "whisper_text": "Hol your dad and your uncle.",
        "corrected_text": "Get your dad and uncles.",
        "ground_truth": "Get your dad and uncles.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "I saw two dog in the park.",
        "corrected_text": "I saw two dogs in the park.",
        "ground_truth": "I saw two dogs in the park.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "There are many car on the street.",
        "corrected_text": "There are many cars on the street.",
        "ground_truth": "There are many cars on the street.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "Bring these book to the library.",
        "corrected_text": "Bring these books to the library.",
        "ground_truth": "Bring these books to the library.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "We need more coworker for this project.",
        "corrected_text": "We need more coworkers for this project.",
        "ground_truth": "We need more coworkers for this project.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "He has a few question to ask.",
        "corrected_text": "He has a few questions to ask.",
        "ground_truth": "He has a few questions to ask.",
        "source_file": "synthetic_targeted"
    },
    {
        "whisper_text": "Look at all those bird flying.",
        "corrected_text": "Look at all those birds flying.",
        "ground_truth": "Look at all those birds flying.",
        "source_file": "synthetic_targeted"
    }
]

# Generate variations to bulk it up to ~150 targeted examples
extra_pairs = []
# Newline templates
newline_templates = [
    ("Please call me tomorrow. I have something to say.", "Please call me\ntomorrow. I have something to say."),
    ("I don't know what you want. Please leave me alone.", "I don't know what you want.\nPlease leave me alone."),
    ("If you see her, tell her I am waiting here.", "If you see her,\ntell her I am waiting here."),
    ("This is the first time I have been here.", "This is the first time\nI have been here."),
    ("We should go now. It is getting late.", "We should go now.\nIt is getting late."),
    ("I am working on a new project. It is very hard.", "I am working on a new project.\nIt is very hard."),
    ("She lives in a small town. You will like it.", "She lives in a small town.\nYou will like it."),
    ("Don't worry about it. Everything is fine.", "Don't worry about it.\nEverything is fine."),
    ("He wants to see you. Can you come?", "He wants to see you.\nCan you come?"),
    ("Let's take a break. We need some rest.", "Let's take a break.\nWe need some rest.")
]
for w, r in newline_templates:
    extra_pairs.append({
        "whisper_text": w,
        "corrected_text": r,
        "ground_truth": r,
        "source_file": "synthetic_targeted"
    })

# Music note templates
music_templates = [
    ("I believe I can fly", "♪ I believe I can fly ♪"),
    ("Yesterday all my troubles seemed so far away", "♪ Yesterday, all my troubles\nseemed so far away ♪"),
    ("Walking in a winter wonderland", "♪ Walking in a winter wonderland ♪"),
    ("Let it go, let it go", "♪ Let it go, let it go ♪"),
    ("Welcome to the jungle", "♪ Welcome to the jungle ♪"),
    ("I hear the drums echoing tonight", "♪ I hear the drums\nechoing tonight ♪"),
    ("Ground control to Major Tom", "♪ Ground control to Major Tom ♪"),
    ("Here comes the sun", "♪ Here comes the sun ♪"),
    ("Under pressure pressing down on me", "♪ Under pressure\npressing down on me ♪"),
    ("Imagine all the people sharing all the world", "♪ Imagine all the people\nsharing all the world ♪")
]
for w, r in music_templates:
    extra_pairs.append({
        "whisper_text": w,
        "corrected_text": r,
        "ground_truth": r,
        "source_file": "synthetic_targeted"
    })

# Plurality templates
plurality_templates = [
    ("She has three cat.", "She has three cats.", "She has three cats."),
    ("I bought five apple.", "I bought five apples.", "I bought five apples."),
    ("There are several student in the room.", "There are several students in the room.", "There are several students in the room."),
    ("They visited many country last year.", "They visited many countries last year.", "They visited many countries last year."),
    ("All my friend are coming tonight.", "All my friends are coming tonight.", "All my friends are coming tonight."),
    ("We need two box of matches.", "We need two boxes of matches.", "We need two boxes of matches."),
    ("He has many book in his library.", "He has many books in his library.", "He has many books in his library."),
    ("I saw three child playing outside.", "I saw three children playing outside.", "I saw three children playing outside."),
    ("Those house are very old.", "Those houses are very old.", "Those houses are very old."),
    ("Give me those key.", "Give me those keys.", "Give me those keys."),
    ("I want to buy some orange.", "I want to buy some oranges.", "I want to buy some oranges."),
    ("How many ticket did you buy?", "How many tickets did you buy?", "How many tickets did you buy?"),
    ("We need more desk in the office.", "We need more desks in the office.", "We need more desks in the office."),
    ("The bird are singing in the tree.", "The birds are singing in the tree.", "The birds are singing in the tree."),
    ("I love to eat banana.", "I love to eat bananas.", "I love to eat bananas."),
    ("She has a few brother.", "She has a few brothers.", "She has a few brothers."),
    ("He has two uncle and three aunt.", "He has two uncles and three aunts.", "He has two uncles and three aunts."),
    ("Get your dad and uncle.", "Get your dad and uncles.", "Get your dad and uncles."),
    ("We have two car.", "We have two cars.", "We have two cars."),
    ("Look at the star in the sky.", "Look at the stars in the sky.", "Look at the stars in the sky.")
]
for w, r, gt in plurality_templates:
    extra_pairs.append({
        "whisper_text": w,
        "corrected_text": r,
        "ground_truth": gt,
        "source_file": "synthetic_targeted"
    })

all_targeted = TARGETED_PAIRS + extra_pairs

# Append to training_pairs.jsonl
input_path = Path.home() / "Downloads" / ".subcache" / "training_pairs.jsonl"
with open(input_path, "a") as f:
    for pair in all_targeted:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"Successfully appended {len(all_targeted)} targeted synthetic pairs to training_pairs.jsonl")
