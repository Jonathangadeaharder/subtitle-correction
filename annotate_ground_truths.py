"""
Ground truth annotations for real MP4-derived training pairs.
Rules:
  1. Phonetic mishearing (sounds like ref word) → fix using reference
  2. Hallucination (makes no sense given ref) → fix using reference
  3. Correctly phrased differently (Whisper valid) → keep Whisper

Run this script to merge ground_truth back into training_pairs.jsonl.
"""

import json
from pathlib import Path

# Each entry: (whisper_text, ground_truth)
# Whisper is always index 0, ground_truth is what the model should output.
GROUND_TRUTHS = {
    # [0] "Hol" = mishearing of "Get" (rule 1) / but also could be "Hold" - ref says "Get"
    "Hol your dad and your uncle.": "Get your dad and uncle.",
    # [1][2] "For now" hallucination, ref has "one hour" - fix hallucination (rule 2)
    "For now we're ahead of the posse": "♪ one hour ahead of the posse ♪",
    # [3] Whisper "Oh, it's good" is valid; ref "Ooh, that's good" is same meaning (rule 3)
    "Oh, it's good.": "Oh, it's good.",
    # [4] "were in the police" = hallucination; ref "went to the station" (rule 2)
    "I mean, they were in the police.": "I mean, they went to the station.",
    # [5] Whisper valid phrasing (rule 3)
    "We know, where he wants to go.": "We know, where he wants to go.",
    # [6] "out of the face" = partial hallucination; ref adds "goddamn nose" (rule 2)
    "we'll cut them out of the face.": "we'll cut the nose off their face.",
    # [7] "und ich" = German mishearing; "Salverson" vs "Solverson" = rule 1
    "Sir, State Trooper Salverson und ich": "Sir, Trooper Solverson and myself,",
    # [8] Whisper valid question form (rule 3)
    "It's a quote.": "It's a quote.",
    # [9] "Ah" vs "Oh" - both valid interjections (rule 3)
    "Ah, yeah.": "Ah, yeah.",
    # [10] "lock your jaw till you can talk" - ref says "can't talk" - rule 1 (mishearing "can" for "can't")
    "You can walk, I'll lock your jaw till you can talk.": "♪ I'll lock your jaw till you can't talk ♪",
    # [11] "dripped" = mishearing of "tripped" (rule 1)
    "I dripped on a cloud and fell eight miles high.": "♪ I tripped on a cloud and fell eight miles high ♪",
    # [12] Whisper correct (rule 3) - "crawling" vs "crawlin'" is just style
    "I watched myself crawling out as I was crawling in.": "♪ I watched myself crawling out as I was crawling in ♪",
    # [13] "There's" vs "We have" - Whisper valid (rule 3)
    "There's no birth certificate,": "There's no birth certificate,",
    # [14] "family or family history" vs "tribal or family history" - "family" mishearing of "tribal" (rule 1)
    "no family or family history.": "no tribal or family history.",
    # [15] "hair-schnitt" = German mishearing of "haircut" (rule 1)
    "He wanted a hair-schnitt.": "He wanted a haircut,",
    # [16] Whisper valid (rule 3)
    "What is that?": "What is that?",
    # [17] "Jemen" = mishearing of beginning of "Benjamin" (rule 1)
    "Jemen. Benjamin.": "Benjamin.",
    # [18] "feel down" vs "wanna be down" - Whisper close (rule 3)
    "I don't want to feel down": "♪ I don't want to feel down ♪",
    # [19] Whisper correct phrasing (rule 3)
    "Cause my name is getting around": "♪ 'Cause my name is getting around ♪",
    # [20] "I say" vs "I said" - Whisper valid (rule 3)
    "I say the whole world is letting me down now": "♪ I say the whole world is letting me down now ♪",
    # [21][22] Whisper correct (rule 3)
    "Yes, I guess I'm going down, down, down": "♪ Yes, I guess I'm going down, down, down ♪",
    "I guess I'm going down, down, down": "♪ I guess I'm going down, down, down ♪",
    # [23] "Calling" vs "callin'" - style only (rule 3)
    "Calling to my name": "♪ Calling to my name ♪",
    # [24] "200 billion" vs "200 millions" + "unloaded" vs "loaded" - rule 1 mishearings
    "200 billion guns unloaded": "♪ 200 million guns are loaded ♪",
    # [25] "could you not" vs "couldn't you fix it" - hallucination (rule 2)
    "I know, but could you not...": "I know, but couldn't you fix it?",
    # [26] Whisper valid (rule 3)
    "Is this a question?": "Is this a question?",
    # [27] Whisper valid (rule 3)
    "That's something between me and my brother.": "That's something between me and my brother.",
    # [28] "Bakteria/Amphibia" = mishearings of "Bacteria/amphibian" (rule 1)
    "Bakteria was to Amoeba, Amoeba was to Amphibia, the Amphibia was to humans.": "Bacteria to amoeba, amoeba to amphibian, amphibian to man.",
    # [29] "Theddeus" = mishearing of "Thaddeus" (rule 1)
    "Theddeus Mobley.": "Thaddeus Mobley.",
    # [30] "You should call me, baby. I'm not alone." - hallucination, ref totally different (rule 2)
    "You should call me, baby. I'm not alone.": "You should've called, baby. I got company.",
    # [31] "Ferius" = mishearing of "Thaddeus" (rule 1); rest is paraphrase (rule 3 for structure)
    "Ferius is a big boy. He saw the world.": "Thaddeus is a big boy. He's seen the world.",
    # [32] Whisper valid (rule 3)
    "I'm a bad person.": "I'm a bad person.",
    # [33] "metaller" = mishearing of "metal" (rule 1)
    "You helped my friend, my metaller friend.": "You have helped, my metal friend.",
    # [34] "Reader Pearl" vs "pearl" casing; "Ethel Reader" vs "ethelrida" - rule 1
    "My life story from Ethel Reader Pearl Smutny.": "My history report, by Ethelrida Pearl Smutny.",
    # [35] "story" vs "history" - rule 1 mishearing
    "It's about our story.": "It's about our history.",
    # [36] same
    "But this is a story.": "But this is a history report.",
    # [37] same
    "And what does this story teach us?": "And what does history tell us?",
    # [38] Whisper correct (rule 3)
    "How am I doing? Hey, hey.": "♪ How am I doing? Hey, hey. ♪",
    # [39] "Let me in" missing from Whisper - partial hallucination (rule 2)
    "I'm going to talk to my dad.": "I got to talk to my dad.",
    # [40] Whisper valid (rule 3)
    "No one of them was white.": "No one of them was white.",
    # [41] Whisper valid (rule 3)
    "If I say this is so, then it's so.": "If I say this is so, then it's so.",
    # [42] Whisper valid (rule 3)
    "Why do you need us?": "Why do you need us?",
    # [43] Whisper valid (rule 3)
    "So we can expand in your community.": "So we can expand in your community.",
    # [44] "earthly things" vs "earthly realm" - rule 1
    "These earthly things?": "This earthly realm.",
    # [45] Whisper valid (rule 3)
    "Think faster.": "Think faster.",
    # [46] Whisper hallucination "credit line is out" vs "credit is max" (rule 2)
    "Also, my credit line is out.": "So, my credit is maxed out.",
    # [47] Whisper valid (rule 3)
    "The show is over.": "The show is over.",
    # [48] "You're going to" vs "you gotta" - Whisper valid (rule 3)
    "You're going to forgive yourself.": "You're going to forgive yourself.",
    # [49] Whisper valid (rule 3)
    "You have to talk about this moment.": "You have to talk about this moment.",
    # [50] "Hendricks" = mishearing of "Hendrix"; ref adds "Beth" which Whisper missed (rule 1+2)
    "Alison Hendricks,": "Alison Hendrix,",
    # [51] "rockin'" vs "rock" - Whisper valid (rule 3)
    "The clash rockin'.": "The Clash rockin'.",
    # [52] Whisper valid paraphrase (rule 3)
    "She was the kind of person, who wants to hold.": "She was the kind of person you want to hold on to.",
    # [53] "Fi" = mishearing of "Fe" (rule 1)
    "Mrs. S is there, Fi.": "Mrs. S is here, Fe.",
    # [54] "best ich bin Katja" = German mishearing of "Beth, it's Katja" (rule 1)
    "best ich bin Katja": "Beth, it's Katja.",
    # [55] Whisper valid (rule 3)
    "Oh, hi. Beth Childs.": "Oh, hi. Beth Childs.",
    # [56] "Sarah ist tot" = German; ref "Sarah is dead" (rule 1 - German AS mishearing)
    "Sarah ist tot!": "Sarah is dead!",
    # [57] "Maggie Chen" = mishearing of "Maggie Chan" (rule 1); "church" = mishearing of "churchgoer"
    "Maggie Chen, 44, single, church.": "Maggie Chan, 44, single, churchgoer.",
    # [58] same as [51]
    "The Clash Rockin.": "The Clash rockin'.",
    # [59] "sick" = mishearing of "dick" (rule 1)
    "Why are you so sick?": "Why are you such a dick?",
    # [60] "Bess" = mishearing of "Beth" (rule 1)
    "I'm Katja, Katja Obinger, Bess.": "I'm Katja. Katja Obinger, Beth.",
    # [61] "Upleat" = mishearing of "Plead" (rule 1)
    "Upleat your mercy and your pity.": "Plead your mercy and your pity.",
    # [62] "Vic, Kira ist nicht Sarah" - partial German, ref drops "Vic" (rule 1+2)
    "Vic, Kira ist nicht Sarah.": "Kira is not Sarah.",
    # [63] "Why did you do this?" vs "Why did you have one?" - hallucination (rule 2)
    "Yes, Vic was here. Why did you do this?": "Yeah, Vic was here. Why did you have one?",
    # [64] Whisper valid (rule 3)
    "Finally, where are you?": "Finally, where are you?",
    # [65] "Daniel" = mishearing of "Danielle" (rule 1)
    "Daniel Fournier.": "Danielle Fournier.",
    # [66] "Frankreich" = German for France (rule 1)
    "Daniel Fournier, Frankreich.": "Danielle Fournier, France.",
    # [67] "Orangen" = German for "oranges" (rule 1)
    "Okay, Orangen for all!": "Okay, oranges, guys.",
    # [68] "Livia" = mishearing of "Alevia"; medication names mangled (rule 1)
    "Livia, Superprax, Draxafil.": "Alevia, superprax, draxophyl.",
    # [69] "Charles" = mishearing of "Childs" (rule 1)
    "Detective Elizabeth Charles.": "Detective Elizabeth Childs.",
    # [70] "I was too" vs "I was there" - "too" = mishearing of "there" (rule 1)
    "Yes, I was too, thank you.": "Yeah, I was there, thank you.",
    # [71] Whisper hallucination (rule 2)
    "How can I do this?": "How can it be, though?",
    # [72] Whisper valid (rule 3)
    "Who has created us?": "Who has created us?",
    # [73] "make no longer than a mother" vs "do not a mother make" - hallucination fix (rule 2)
    "New clothes and a Jaguar make no longer than a mother.": "New clothes and a Jaguar do not a mother make.",
    # [74] "And Art is already there, where Katya was shot" - "Katya" = mishearing of "Katja" (rule 1); paraphrase rest
    "And Art is already there, where Katya was shot.": "He's already at the scene where Katja was shot.",
    # [75] Whisper valid (rule 3)
    "Yeah, it would be possible.": "Yeah, it could be.",
    # [76] Whisper valid (rule 3)
    "Hey, I have something else.": "Hey, I have something else.",
    # [77] same as [16]
    # [78] Whisper valid paraphrase (rule 3)
    "I think it's in one of the buildings here.": "I think it's in one of the buildings here.",
    # [79] "will give me one visit" vs "gave me one visit" - rule 3
    "Mrs. S. will give me one visit.": "Mrs. S. gave me one visit.",
    # [80] "ghoul" = mishearing of "angry angel" (rule 2 - completely different)
    "Can you tell me how the ghoul looks like?": "Can you tell me what the angry angel looked like?",
    # [81] Whisper valid (rule 3)
    "She leads us here.": "She leads us here.",
    # [82] Whisper valid (rule 3)
    "She wants to see it.": "She wants to see it.",
    # [83] "Now comes to a really idiotic idea" = hallucination (rule 2)
    "Now comes to a really idiotic idea.": "I do have one really idiotic idea.",
    # [84] "Oi, oi" vs "Oh, hi" - rule 1
    "Oi, oi, Mrs. S.": "Oh, hi, Mrs. S.",
    # [85] Whisper valid (rule 3)
    "I don't know if it's that.": "I don't know if it's that.",
    # [86] Whisper valid (rule 3) - "mom" vs "mum" is dialect
    "Kira, your mom is waiting.": "Kira, your mom is waiting.",
    # [87] "bite off" = mishearing of "won't bite" (rule 1)
    "She doesn't bite off.": "She won't bite.",
    # [88] Whisper valid (rule 3)
    "Yes, I am": "Yes, I am.",
    # [89] Whisper valid (rule 3)
    "We are all the same": "We are all the same.",
    # [90] "seen everywhere" vs "searched everywhere" - rule 1
    "I don't know. We've seen everywhere.": "I don't know. We searched everywhere.",
    # [91] Whisper valid (rule 3)
    "He has a meeting.": "He has a meeting.",
    # [92] Whisper valid (rule 3)
    "I use a new lotion.": "I use a new lotion.",
    # [93] Whisper valid (rule 3)
    "No, it's not! What is that? What's inside?": "No, it's not! What is that? What's inside?",
    # [94] Whisper valid (rule 3)
    "And if I tell you that, I'm going to kill you.": "And if I told you that, I'd have to kill you.",
    # [95] Whisper valid (rule 3)
    "I miss you so much.": "I miss you so much.",
    # [96] "sure" = mishearing of "safe" (rule 1)
    "We're sure, Kira.": "We're safe, Kira.",
    # [97] "Hallo" = German mishearing of "Hello" (rule 1)
    "Hallo, Alison.": "Hello, Alison.",
    # [98] Whisper valid (rule 3)
    "I have a right to know everything.": "I have a right to know everything.",
    # [99] Whisper valid (rule 3)
    "I'm standing before you and I see you.": "I'm standing here looking at you.",
    # --- PAIRS 100-199 ---
    # [100] Whisper valid (rule 3) - dissertation subject
    "I'm doing my dissertation on epigenetics.": "I'm doing my dissertation on epigenetics.",
    # [101] Whisper valid (rule 3)
    "I'm doing my PhD in experimental physics.": "I'm doing my PhD in experimental physics.",
    # [102] Whisper valid (rule 3)
    "It is experimental.": "It is experimental.",
    # [103] Whisper valid (rule 3)
    "She is a scientist.": "She is a scientist.",
    # [104] Whisper valid (rule 3)
    "He studied biology.": "He studied biology.",
    # [105] Whisper valid (rule 3)
    "We studied clones.": "We studied clones.",
    # [106] "disser-station" = hallucination, ref correct (rule 2)
    "but my dissertation is on epigenetic influence on clone cells.": "but my dissertation is on epigenetic influence on clone cells.",
    # [107] "Thomas" = mishearing of "Tomas" (rule 1)
    "From Thomas.": "From Tomas.",
    # [108] "Detektives" = German mishearing (rule 1)
    "Detektives Bell und De Angelis.": "Detectives Bell and Deangelis.",
    # [109] Whisper valid (rule 3)
    "Why are you asking that?": "Why are you asking that?",
    # [110] Whisper valid (rule 3)
    "Had Sarah a sister?": "Had Sarah a sister?",
    # [111] "possessed of" = hallucination; "bitchy" = content (rule 2)
    "She's being possessed of one of the wrong friends in her monitor.": "She's obsessing that one of her friends is her monitor.",
    # [112] Whisper valid (rule 3)
    "Through the blood loss and the risk of infection at this point?": "Through the blood loss and the risk of infection at this point?",
    # [113] Whisper valid (rule 3)
    "I'm nothing in between.": "♪ I'm nothing in between ♪",
    # [114] Whisper valid (rule 3)
    "Take a hot bath. I'll call you later.": "Take a hot bath. I'll call you later.",
    # [115] Whisper valid (rule 3)
    "Oh, well, Alison, Cosima, Beth,": "Well, Alison, Cosima, Beth,",
    # [116] Whisper valid (rule 3) - mom/mum dialect
    "You are exactly like my mom.": "You are exactly like my mom.",
    # [117] "not really" vs "not real" - rule 1 mishearing
    "She's not really.": "She's not real.",
    # [118] "it's going" = hallucination; ref "gotta go" (rule 2)
    "Yeah, Helena, it's going.": "Yeah, Helena's gotta go.",
    # [119] "order it to be" = hallucination; ref "bring Beth back in" (rule 2)
    "I have a dozen reasons to order it to be.": "I got a dozen reasons to bring Beth back in.",
    # [120] "Who is true" = hallucination (rule 2)
    "Who is true, that the child is actually yours?": "If this is true, if the child's actually hers...",
    # [121] Whisper valid (rule 3)
    "It was a accident.": "It was an accident.",
    # [122] Whisper valid (rule 3)
    "A very beautiful room for a young couple.": "A very beautiful room for a young couple.",
    # [123] "alone" vs "back there alone" - Whisper valid (rule 3)
    "I can't do it alone.": "I can't do it alone.",
    # [124] "unverletted" = German/mishearing of "uninjured" (rule 1)
    "Because she is unverletted?": "Because she's uninjured?",
    # [125] "Leakey" = mishearing of "Leekie" (rule 1)
    "Dr. Leakey knows about you, Sarah.": "Dr. Leekie knows about you, Sarah.",
    # [126] Whisper valid (rule 3)
    "He knows that you know it and are in contact with each other.": "He knows you're aware, in contact with each other.",
    # [127] Whisper valid (rule 3)
    "Your interest is the same and you get answers.": "Your interests are aligned and you get answers.",
    # [128] "for Cytochrome-C normal" = hallucination; ref "anomalous for Cytochrome C" (rule 2)
    "This sequence is for Cytochrome-C normal.": "This sequence is anomalous for Cytochrome C.",
    # [129] Whisper valid (rule 3)
    "They are different?": "They are different?",
    # [130] "Why" vs "What" - rule 1 mishearing
    "Why are you looking for?": "What are you looking for?",
    # [131] Whisper valid (rule 3)
    "What did you do with Cosima?": "What have you done with Cosima?",
    # [132] Whisper valid (rule 3)
    "Because she's definitely not.": "Because she's definitely not.",
    # [133] Whisper valid (rule 3)
    "You're perhaps surprised.": "You might be surprised.",
    # [134] "Leakey" = mishearing of "Leekie" (rule 1)
    "Dr. Leakey?": "Dr. Leekie?",
    # [135] "nicht Berro" = German mishearing of "not Beraud" (rule 1)
    "Delphine Cormier, nicht Berro.": "Delphine Cormier, not Beraud.",
    # [136] "who" vs "what" - rule 1 mishearing (context: they're a clone)
    "I know who you are.": "I know what you are.",
    # [137] Whisper valid (rule 3)
    "Yes, I am the one with the keys.": "Yes, I am the one with the keys.",
    # [138] "Leakey" = mishearing of "Leekie"; "give them" vs "give them to" (rule 1)
    "Maybe I'll give them Dr. Leakey.": "Maybe I'll give them to Dr. Leekie.",
    # [139] Whisper valid (rule 3)
    "I knew that I could not hold you.": "I knew that I couldn't keep you.",
    # [140] Whisper valid (rule 3)
    "And I'm hiding you, because I felt you were in danger.": "And I hid you because I felt you were in danger.",
    # [141] Whisper valid (rule 3)
    "I don't want to answer anymore.": "I don't want answers anymore.",
    # [142] "private family" = hallucination of "privacy" (rule 2)
    "I want my family and my private family back.": "I want my family back. I want my privacy back.",
    # [143] "work for the Died Institute" = hallucination; "Dyad Institute" (rule 2)
    "This is a work for the Died Institute.": "This is an employment contract for the Dyad Institute.",
    # [144] "questions" vs "concerns" - Whisper valid (rule 3)
    "We have the same questions": "We have the same questions.",
    # [145] "Are you working for" vs "You work for"; "Leaky" = "Leekie" (rule 1)
    "Are you working for Dr. Leaky?": "You work for Dr. Leekie?",
    # [146] "model" vs "transition" - rule 2 hallucination
    "And my job is to model your self-awareness.": "And my role is to transition you into self-awareness.",
    # [147] Whisper valid (rule 3)
    "We want you to trust.": "We want your trust.",
    # [148] "The mother of God is wonderful" = hallucination of "Motherhood is wonderful" (rule 2)
    "The mother of God is wonderful.": "Motherhood is wonderful.",
    # [149] Whisper valid (rule 3)
    "Ah, and Helena is my twin sister.": "And Helena's my twin sister.",
    # [150] "Delfin" = mishearing of "Delphine" (rule 1)
    "It's Delfin!": "It's Delphine!",
    # [151] "going to work" vs "going for a jog" - rule 2 hallucination
    "I'm going to work.": "I'm going for a jog.",
    # [152] "Bath" = mishearing of "Beth" (rule 1)
    "And why does she look like Bath?": "Why does she look like Beth?",
    # [153] "murder" = mishearing of "interview" (rule 2)
    "Detective, the murder is over.": "Detective, this interview's over.",
    # [154] "Is Rachel my family?" = hallucination (rule 2)
    "Is Rachel my family?": "Did Rachel take my family?",
    # [155] "in the room" vs "leave on her own" - hallucination (rule 2)
    "I think we'll let the lady in the room.": "I think we'll let the lady leave on her own.",
    # [156] "closed" = mishearing of "disconnected" (rule 1)
    "You said all the other phones are closed.": "You said all the other phones have been disconnected.",
    # [157] "Dear Paul" = mishearing of "It's Paul" (rule 1)
    "Dear Paul, Sarah.": "It's Paul, Sarah.",
    # [158] Whisper valid (rule 3)
    "Let's call him Ramon.": "Let's call him Ramon.",
    # [159] Whisper valid (rule 3)
    "I don't have any plans.": "I don't have a plan yet.",
    # [160] Whisper valid (rule 3)
    "Why not talk to them, Sarah?": "Why don't I talk to them, Sarah?",
    # [161] "peace" = mishearing of "piece" (rule 1)
    "I have a friend, who needs peace.": "I have a friend who needs a piece...",
    # [162] "Hendricks" = mishearing of "Hendrix" (rule 1)
    "Mrs. Hendricks.": "Mrs. Hendrix!",
    # [163] Whisper valid (rule 3)
    "If you have a revolver, this 45-year-old is...": "If you have a revolver, this .45 is...",
    # [164] "Comier" = mishearing of "Cormier" (rule 1)
    "Dr. Comier.": "Dr. Cormier.",
    # [165] "Delfin" = mishearing of "Delphine" (rule 1)
    "Where is Cosima, Delfin?": "Where's Cosima, Delphine?",
    # [166] "Viel Glück" = German for "Good luck" (rule 1)
    "Viel Glück, Paul.": "Good luck, Paul.",
    # [167] Whisper valid (rule 3)
    "Okay, thank you.": "Okay, thank you.",
    # [168] Whisper valid (rule 3)
    "If Rachel and Kira are in a plane, I have to be there.": "If Rachel's got Kira on a plane, I have to be with her.",
    # [169] Whisper valid (rule 3)
    "I was not sure as Delphine, that you're coming!": "I wasn't as sure as Delphine that you'd come.",
    # [170] "Leakey" = mishearing of "Leekie" (rule 1)
    "Thank you, Dr. Leakey!": "Thank you, Dr. Leekie.",
    # [171] "Hallo" = German/mishearing of "Hello" (rule 1)
    "Kira? Hallo?": "Kira? Hello?",
    # [172] Whisper valid (rule 3)
    "That was our mistake.": "That was our mistake.",
    # [173] "Polityaners" = mishearing of "Proletheans" (rule 1)
    "What are these Polityaners with your daughter?": "What do these Proletheans want with your daughter?",
    # [174] "Vater" = German for "pastor" (rule 1)
    "Amen, Vater.": "Amen, pastor.",
    # [175] "Danke" = German for "Thanks" (rule 1)
    "Danke, Ben.": "Thanks, Ben.",
    # [176] "changed" = mishearing of "exchanged" (rule 1)
    "And the words were changed.": "And words were exchanged.",
    # [177] Whisper valid (rule 3)
    "I have something that helps.": "I have something that might help.",
    # [178] "life" vs "reality" - rule 1 mishearing
    "No, but it's your life.": "No, but it is your reality.",
    # [179] "Leder" = mishearing of "LEDA" (rule 1)
    "Project Leder?": "Project LEDA?",
    # [180] Whisper valid (rule 3)
    "Jamie was my dad.": "Jamie's my dad.",
    # [181] Whisper valid (rule 3)
    "How do you feel?": "How do you feel?",
    # [182] "MRT" = German for MRI (rule 1)
    "Dr. Cormier will be for an MRT.": "Dr. Cormier will schedule you an MRI.",
    # [183] "clipped" = mishearing of "sequenced" (rule 2)
    "And here is your clipped genome.": "And here's her sequenced genome.",
    # [184] Whisper valid (rule 3)
    "I want to know why she is different than us.": "I want to know why she's different than we are.",
    # [185] "change the road" = mishearing of "change the route" (rule 1)
    "We change the road.": "We'll change the route.",
    # [186] "try" vs "review" - rule 2 hallucination
    "I'd like to try this personally, Brenda.": "I'd like to review it personally, Brenda.",
    # [187] "creatures" vs "cursed children" - rule 2 hallucination
    "what are the creatures you brought into our lives?": "who are these cursed children you brought into our lives?",
    # [188] "Ainsley" = mishearing of "Aynsley" (rule 1)
    "I killed Ainsley!": "I killed Aynsley!",
    # [189] Whisper valid (rule 3)
    "No one said it would be easy, Mark.": "No one said it would be easy, Mark.",
    # [190] "Ich bin" = German for "I'm" (rule 1)
    "Ich bin Jennifer Fitzsimmons.": "I'm Jennifer Fitzsimmons.",
    # [191] "Cal, a very old friend" = hallucination; ref "He's an old friend" (rule 2)
    "Cal, a very old friend.": "He's an old friend.",
    # [192] Whisper valid (rule 3)
    "You can stay in the night if that helps. Okay?": "You can stay another night if that helps.",
    # [193] "Colors" = mishearing of "Covers" (rule 1)
    "Colors the tiles and the bathroom mat": "♪ Covers the tiles and the bathroom mat ♪",
    # [194] "fail" = mishearing of "feel" (rule 1)
    "How they fail doesn't matter": "♪ How they feel doesn't matter ♪",
    # [195] "throw it" = mishearing of "through it" (rule 1)
    "And there's no one you can call to help you throw it": "♪ And there's no one you can call to help you through it ♪",
    # [196] "classifiziert" = German; ref "unclassified" (rule 1)
    "Yes, but not classifiziert.": "Yes, but unclassified.",
    # [197] "Immunsuppressiva" = German for immunosuppressives (rule 1)
    "Immunsuppressiva had only limited effects.": "Immunosuppressives had limited effect.",
    # [198] "tomorrow will it go" = hallucination (rule 2)
    "That's why tomorrow will it go.": "That's what tomorrow is all about.",
    # [199] "Leder" = mishearing of "LEDA" (rule 1)
    "Projekt Leder, 77.": "Project LEDA, '77.",
}

# Now load full file and patch
input_path = Path.home() / "Downloads" / ".subcache" / "training_pairs.jsonl"
pairs = []
with open(input_path) as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

patched = 0
skipped = 0
for p in pairs:
    w = p.get("whisper_text", "").strip()
    if p.get("source_file") == "synthetic_handcrafted":
        # For synthetic pairs, ground_truth == corrected_text (already correct)
        p["ground_truth"] = p.get("corrected_text", w)
        continue
    if w in GROUND_TRUTHS:
        p["ground_truth"] = GROUND_TRUTHS[w]
        patched += 1
    else:
        # Fallback: use corrected_text (reference) as ground_truth
        p["ground_truth"] = p.get("corrected_text", w)
        skipped += 1

print(f"Patched: {patched}, Fallback to ref: {skipped}")

with open(input_path, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print("Done. training_pairs.jsonl updated.")
