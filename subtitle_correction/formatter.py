import re

def restore_formatting(corrected: str, reference: str) -> str:
    """
    Post-processes the model's corrected text output by mapping it back into the
    structural formatting (line breaks, music symbols, capitalization, and punctuation)
    defined in the reference text.
    """
    # 1. Detect if the reference had music symbols
    has_music_start = reference.strip().startswith("♪")
    has_music_end = reference.strip().endswith("♪")
    
    # 2. Extract word lists
    # We clean the reference to match the raw words
    ref_clean = reference.replace("♪", "").strip()
    ref_lines = ref_clean.split("\n")
    
    ref_words_by_line = [line.strip().split() for line in ref_lines]
    ref_words = [w for line in ref_words_by_line for w in line]
    
    corrected_clean = corrected.replace("♪", "").replace("\n", " ").strip()
    corr_words = corrected_clean.split()
    
    if not corr_words:
        return corrected
        
    if not ref_words:
        # Fallback to model output if reference is empty
        res = corrected
        if has_music_start and not res.startswith("♪"):
            res = "♪ " + res
        if has_music_end and not res.endswith("♪"):
            res = res + " ♪"
        return res

    # 3. Simple alignment mapping: match corrected words to reference words
    # If the lengths match exactly, we can map word-for-word.
    # Otherwise, we use a simple heuristic to preserve line-breaks at roughly the same position.
    mapped_words = []
    
    # Pre-clean punctuation helper
    def clean_word(w):
        return re.sub(r"[^\w\s]", "", w.lower()).strip()
    
    for i, cw in enumerate(corr_words):
        # Look for matching reference word close to the same index
        best_ref_idx = None
        best_dist = 999
        
        # Search window
        search_start = max(0, i - 3)
        search_end = min(len(ref_words), i + 4)
        
        for r_idx in range(search_start, search_end):
            rw = ref_words[r_idx]
            if clean_word(cw) == clean_word(rw):
                dist = abs(i - r_idx)
                if dist < best_dist:
                    best_dist = dist
                    best_ref_idx = r_idx
                
        # If we found a direct match, copy its casing and punctuation
        if best_ref_idx is not None:
            rw = ref_words[best_ref_idx]
            # Preserve capitalization of reference word
            if rw and rw[0].isupper():
                cw = cw[0].upper() + cw[1:] if cw else ""
            elif rw and rw[0].islower():
                cw = cw[0].lower() + cw[1:] if cw else ""
                
            # Copy leading/trailing punctuation
            leading_punc = ""
            trailing_punc = ""
            
            # Leading punc (e.g. quotes, dashes)
            m_lead = re.match(r"^([^\w\s]+)", rw)
            if m_lead:
                leading_punc = m_lead.group(1).replace("♪", "")
                
            # Trailing punc
            m_trail = re.search(r"([^\w\s]+)$", rw)
            if m_trail:
                trailing_punc = m_trail.group(1).replace("♪", "")
                
            # Apply to corrected word
            cw_stripped = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", cw)
            cw = leading_punc + cw_stripped + trailing_punc
        else:
            # No match found in the close neighborhood; preserve casing/punc of nearest ref if first/last
            if i == 0 and ref_words[0][0].isupper():
                cw = cw.capitalize()
            if i == len(corr_words) - 1:
                # Add trailing punctuation from last word of ref
                m_trail = re.search(r"([^\w\s]+)$", ref_words[-1])
                if m_trail:
                    cw = cw.rstrip(".,!?\"';:♪") + m_trail.group(1).replace("♪", "")
                    
        mapped_words.append(cw)
        
    # 4. Reconstruct line breaks (\n)
    # We place line breaks at the same relative word boundaries
    final_tokens = []
    
    # Calculate word budget per line based on reference
    ref_line_word_counts = [len(line) for line in ref_words_by_line]
    
    current_word_idx = 0
    for line_idx, word_count in enumerate(ref_line_word_counts):
        if word_count == 0:
            continue
        # Allocate words to this line
        # If it's the last line, take all remaining words
        if line_idx == len(ref_line_word_counts) - 1:
            line_words = mapped_words[current_word_idx:]
        else:
            # Scale proportionally if the model output length changed
            ratio = len(mapped_words) / len(ref_words)
            target_count = max(1, int(round(word_count * ratio)))
            line_words = mapped_words[current_word_idx:current_word_idx + target_count]
            current_word_idx += len(line_words)
            
        if line_words:
            final_tokens.append(" ".join(line_words))
            
    result = "\n".join(final_tokens)
    
    # 5. Re-apply music symbols
    if has_music_start:
        result = "♪ " + result.lstrip("♪ ")
    if has_music_end:
        result = result.rstrip(" ♪") + " ♪"
        
    return result
