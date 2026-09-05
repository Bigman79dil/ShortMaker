import os
import sys
import json
import time
import math
import re
import warnings
from typing import List, Dict

import ffmpeg
import whisper
import ollama

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================

INPUT_VIDEO = "input.mp4"
OUTPUT_DIR = "./output_shorts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ollama model
LLM_MODEL = "mistral-small3.2"

# Whisper model
WHISPER_MODEL = "medium"

# Ignore intro/outro
IGNORE_START = 60.0
IGNORE_END = 60.0

# Clip lengths
MIN_CLIP_LEN = 9.0
TARGET_CLIP_LEN = 12.0
MAX_CLIP_LEN = 17.0

# Output
TOP_CLIPS_TO_EXPORT = 3
MAX_CANDIDATES_TO_LLM = 10

# Titles
TITLE_MAX_LEN = 45
TITLE_CONTEXT_BEFORE = 60.0
TITLE_CONTEXT_AFTER = 90.0

# ==========================================================
# HYPE WORDS
# ==========================================================

HYPE_PHRASES = {
    "oh my god": 5,
    "oh my gosh": 5,
    "no way": 5,
    "let's go": 5,
    "lets go": 5,
    "bro": 2,
    "wait": 2,
    "what": 1,
    "this is crazy": 5,
    "this is insane": 5,
    "i did it": 4,
    "i did that": 4,
    "i'm on fire": 4,
    "im on fire": 4,
    "i take that back": 5,
    "this is impossible": 5,
    "impossible": 4,
    "i can't": 3,
    "what is this": 3,
    "no checkpoint": 5,
    "finally": 3,
}

BORING_STARTS = {
    "welcome back",
    "today i",
    "today i'm",
    "ladies and gentlemen",
    "in this video",
    "so today",
}

# ==========================================================
# PROGRESS BAR
# ==========================================================

def progress(stage, current, total, start_time):

    pct = (current / total) * 100

    bar_len = 30
    filled = int(bar_len * current / total)

    bar = "█" * filled + "░" * (bar_len - filled)

    elapsed = time.time() - start_time

    if current == 0:
        eta = "--:--"
    else:
        remaining = elapsed / current * (total - current)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        eta = f"{mins:02}:{secs:02}"

    sys.stdout.write(
        f"\r{stage} [{bar}] {pct:.1f}% | {int(elapsed)}s | ETA {eta}"
    )
    sys.stdout.flush()

# ==========================================================
# LLM HELPERS
# ==========================================================

def chat(prompt, temperature=0):

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": temperature
        }
    )

    return response["message"]["content"].strip()


def extract_json(text):

    try:
        return json.loads(text)

    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)

    if m:

        try:
            return json.loads(m.group())

        except Exception:
            pass

    return None

# ==========================================================
# TRANSCRIPT HELPERS
# ==========================================================

def transcript_to_text(segments):

    lines = []

    for s in segments:

        lines.append(
            f"[{s['start']:.2f}-{s['end']:.2f}] {s['text'].strip()}"
        )

    return "\n".join(lines)


def clip_text(segments, start, end):

    text = []

    for s in segments:

        if s["start"] >= start and s["end"] <= end:
            text.append(s["text"].strip())

    return " ".join(text)


def clip_context(
    segments,
    start,
    end,
    before=TITLE_CONTEXT_BEFORE,
    after=TITLE_CONTEXT_AFTER
):

    s = max(0, start - before)
    e = end + after

    out = []

    for seg in segments:

        if seg["end"] >= s and seg["start"] <= e:

            out.append(
                f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}"
            )

    return "\n".join(out)

# ==========================================================
# TRANSCRIBE VIDEO
# ==========================================================

def transcribe():

    print("Loading Whisper...")

    model = whisper.load_model(WHISPER_MODEL)

    print("Transcribing video...")

    result = model.transcribe(
        INPUT_VIDEO,
        temperature=0,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
        verbose=False
    )

    segments = result["segments"]

    duration = result.get("duration")

    if not duration:
        duration = segments[-1]["end"]

    return segments, duration
# ==========================================================
# MOMENT SCORING SYSTEM
# ==========================================================

def score_text(text):

    """
    Scores transcript sections based on:
    - excitement
    - reactions
    - possible payoff moments
    - natural speech patterns
    """

    low = text.lower()

    score = 0


    # Hype phrases
    for phrase, points in HYPE_PHRASES.items():

        if phrase in low:
            score += points


    # Excitement punctuation
    score += min(text.count("!"), 3)


    # Question/reaction moments
    if "?" in text:
        score += 2


    # Strong reaction words
    reaction_words = [
        "crazy",
        "insane",
        "amazing",
        "beautiful",
        "perfect",
        "failed",
        "almost",
        "close",
        "saved",
        "destroyed",
        "broken",
        "secret",
        "hidden"
    ]

    for word in reaction_words:

        if word in low:
            score += 2


    # Penalize intros/setup
    for start in BORING_STARTS:

        if low.startswith(start):
            score -= 5


    # Too short/too long speech
    words = len(text.split())

    if words < 5:
        score -= 2

    if words > 45:
        score -= 1


    return score



def calculate_segment_scores(segments):

    """
    Gives every transcript segment a hype score.
    Uses nearby segments too because the best moments
    usually build up.
    """

    scores = []

    for i, seg in enumerate(segments):

        score = score_text(seg["text"])


        # Look around the segment
        for x in range(
            max(0, i - 2),
            min(len(segments), i + 3)
        ):

            if x != i:

                score += score_text(
                    segments[x]["text"]
                ) * 0.35


        scores.append(score)


    return scores



# ==========================================================
# BUILD SHORT WINDOWS
# ==========================================================

def get_window(
    segments,
    center,
    target=TARGET_CLIP_LEN
):

    start_index = center
    end_index = center


    def length():

        return (
            segments[end_index]["end"]
            -
            segments[start_index]["start"]
        )


    # Expand around the important moment

    while length() < target:

        left_possible = start_index > 0

        right_possible = (
            end_index < len(segments)-1
        )


        if not left_possible and not right_possible:
            break


        # Prefer adding before the moment
        # because context helps retention

        if left_possible:

            start_index -= 1

        elif right_possible:

            end_index += 1



        if length() >= MAX_CLIP_LEN:
            break



    # Trim if too long

    while length() > MAX_CLIP_LEN:

        if start_index < center:

            start_index += 1

        elif end_index > center:

            end_index -= 1

        else:

            break



    return (
        segments[start_index]["start"],
        segments[end_index]["end"]
    )



# ==========================================================
# CREATE POSSIBLE CLIPS
# ==========================================================

def find_candidates(segments):

    print("\nFinding interesting moments...")


    scores = calculate_segment_scores(
        segments
    )


    ranked = sorted(
        range(len(scores)),
        key=lambda x: scores[x],
        reverse=True
    )


    candidates = []


    for index in ranked:


        if len(candidates) >= MAX_CANDIDATES_TO_LLM:
            break


        start, end = get_window(
            segments,
            index
        )


        duration = end - start


        # Ignore intro/outro

        if start < IGNORE_START:
            continue


        if end > segments[-1]["end"] - IGNORE_END:
            continue


        if duration < MIN_CLIP_LEN:
            continue



        text = clip_text(
            segments,
            start,
            end
        )


        candidate = {

            "start": round(start, 2),

            "end": round(end, 2),

            "duration": round(duration, 2),

            "score": round(
                scores[index],
                2
            ),

            "text": text

        }



        # Avoid almost identical clips

        duplicate = False


        for old in candidates:

            overlap_start = max(
                old["start"],
                candidate["start"]
            )

            overlap_end = min(
                old["end"],
                candidate["end"]
            )


            overlap = (
                max(0, overlap_end - overlap_start)
                /
                min(
                    old["duration"],
                    candidate["duration"]
                )
            )


            if overlap > 0.6:

                duplicate = True
                break



        if not duplicate:

            candidates.append(
                candidate
            )


    print(
        f"Found {len(candidates)} possible moments"
    )


    return candidates
# ==========================================================
# LLM CLIP RANKING
# ==========================================================

def rank_clips(candidates, segments):

    if not candidates:
        return []


    data = ""

    for i, clip in enumerate(candidates):

        context = clip_context(
            segments,
            clip["start"],
            clip["end"],
            before=30,
            after=30
        )

        data += f"""
CANDIDATE {i+1}

TIME:
{clip['start']}s - {clip['end']}s

AUTOMATIC SCORE:
{clip['score']}

CLIP:
{clip['text']}

CONTEXT:
{context}

---------------------
"""


    prompt = f"""
You are an expert YouTube Shorts editor.

Pick the best {TOP_CLIPS_TO_EXPORT} clips.

Judge them by:

1. First 3 seconds:
Does it instantly make someone want to keep watching?

2. Payoff:
Does something interesting actually happen?

3. Context:
Would a random viewer understand what is happening?

4. Replay value:
Would someone watch it again?

5. Comment potential:
Would people react or discuss it?

Avoid:
- introductions
- walking around
- explaining before something happens
- boring setup

Return ONLY this format:

CLIP 1: START: number | END: number
CLIP 2: START: number | END: number
CLIP 3: START: number | END: number


{data}
"""


    answer = chat(
        prompt,
        temperature=0
    )


    matches = re.findall(
        r"START:\s*([\d.]+)\s*\|\s*END:\s*([\d.]+)",
        answer
    )


    selected = []


    for start, end in matches:

        selected.append(
            {
                "start": float(start),
                "end": float(end)
            }
        )


    return selected[:TOP_CLIPS_TO_EXPORT]



# ==========================================================
# TITLE GENERATION
# ==========================================================


def generate_title_options(
    clip_text,
    context
):


    prompt = f"""
You are making YouTube Shorts titles for gaming videos.

Create 15 possible titles.

The titles MUST:

- be based on the actual moment
- make people curious
- sound human
- use simple words
- not be cringe
- not explain the whole clip
- work without knowing the creator
- Use all Caps Sometimes
- Add !!!,!,?,... if it is needed
- if you do use full caps then make sure to add either !!! or ! but if its a question in full caps then use !? or !?!?!?


Good examples:

"HOW DID HE GET ME!!!"
"Why Did I Do That..."
"THIS WAS IMPOSSIBLE!"
"I ACTUALLY DID IT!!!"
"Why Did I Try This?"


Bad examples:

"Playing Minecraft Parkour"
"Trying A New Map"
"Crazy Gaming Moment"


Rules:

- Under 45 characters
- No emojis
- No fake clickbait
- No random words


Return ONLY JSON:

{{
"title_options":[
"title 1",
"title 2"
]
}}


CLIP:
{clip_text}


CONTEXT:
{context}

"""


    result = chat(
        prompt,
        temperature=0.4
    )


    data = extract_json(result)


    if isinstance(data, dict):

        return data.get(
            "title_options",
            []
        )


    return []



# ==========================================================
# TITLE QUALITY SCORING
# ==========================================================


def title_score(title):

    if not title:
        return -999


    score = 0

    low = title.lower()

    words = len(title.split())


    # Good length

    if 3 <= words <= 8:
        score += 3

    elif words > 10:
        score -= 3



    # Curiosity

    curiosity_words = [
        "why",
        "how",
        "what",
        "actually",
        "almost",
        "finally",
        "secret",
        "never",
        "thought",
        "shouldn't",
        "didn't"
    ]


    for word in curiosity_words:

        if word in low:

            score += 2



    # Reaction style

    reaction_words = [
        "impossible",
        "crazy",
        "insane",
        "failed",
        "saved",
        "won",
        "lost",
        "clutch",
        "broken"
    ]


    for word in reaction_words:

        if word in low:

            score += 2



    # Remove boring titles

    bad = [
        "playing",
        "minecraft",
        "gameplay",
        "video",
        "episode",
        "short",
        "trying"
    ]


    for word in bad:

        if word in low:

            score -= 2



    if len(title) <= TITLE_MAX_LEN:

        score += 2


    return score



def choose_best_title(
    titles
):

    if not titles:

        return "I Can't Believe This Happened"



    ranked = sorted(
        titles,
        key=title_score,
        reverse=True
    )


    best = ranked[0]


    # Ask model to make final choice
    # between the strongest options


    top = ranked[:5]


    prompt = f"""
Choose the best YouTube Shorts title.

Rules:
- highest curiosity
- sounds natural
- accurate
- makes people click
- not cringe

Return ONLY the title.

OPTIONS:

{json.dumps(top, indent=2)}
"""


    answer = chat(
        prompt,
        temperature=0
    )


    answer = answer.strip().replace(
        '"',
        ""
    )


    if len(answer) <= TITLE_MAX_LEN:

        return answer


    return best



def create_title(
    clip,
    segments
):

    context = clip_context(
        segments,
        clip["start"],
        clip["end"]
    )


    text = clip_text(
        segments,
        clip["start"],
        clip["end"]
    )


    options = generate_title_options(
        text,
        context
    )


    return choose_best_title(
        options
    )
# ==========================================================
# VIDEO EXPORT
# ==========================================================

def export_short(
    start,
    end,
    filename
):

    duration = end - start

    if duration <= 0:
        return


    print(
        f"\nCutting {filename} ({duration:.1f}s)"
    )


    input_file = ffmpeg.input(
        INPUT_VIDEO,
        ss=start,
        t=duration
    )


    video = input_file.video
    audio = input_file.audio


    # Vertical Shorts format

    video = (
        video
        .filter(
            "scale",
            1080,
            -1
        )
        .filter(
            "pad",
            1080,
            1920,
            "(ow-iw)/2",
            "(oh-ih)/2",
            "black"
        )
    )


    (
        ffmpeg
        .output(
            video,
            audio,
            filename,
            vcodec="libx264",
            preset="slow",
            crf=18,
            acodec="aac",
            audio_bitrate="192k",
            r=60,
            loglevel="quiet"
        )
        .global_args(
            "-nostdin"
        )
        .overwrite_output()
        .run()
    )



# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_log(
    clips
):

    path = os.path.join(
        OUTPUT_DIR,
        "UPLOADS_READY.txt"
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:


        f.write(
            "=== SHORTS OUTPUT LOG ===\n\n"
        )


        for i, clip in enumerate(clips):

            f.write(
                f"SHORT {i+1}\n"
            )

            f.write(
                f"TITLE: {clip['title']}\n"
            )

            f.write(
                f"START: {clip['start']}\n"
            )

            f.write(
                f"END: {clip['end']}\n"
            )

            f.write(
                "\nCLIP TEXT:\n"
            )

            f.write(
                clip["text"]
            )

            f.write(
                "\n\n"
            )

            f.write(
                "=" * 50
            )

            f.write(
                "\n\n"
            )


    print(
        "\nSaved:",
        path
    )



# ==========================================================
# MAIN
# ==========================================================

def main():


    total_start = time.time()


    print(
        "🎬 Stage 1/4: Transcription"
    )


    segments, duration = transcribe()


    print(
        f"\nTranscript loaded: {len(segments)} segments"
    )


    with open(
        os.path.join(
            OUTPUT_DIR,
            "transcript.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            segments,
            f,
            indent=2
        )


    print(
        "\n🔥 Stage 2/4: Finding moments"
    )


    candidates = find_candidates(
        segments
    )


    if not candidates:

        print(
            "No candidates found."
        )

        return



    print(
        "\n🧠 Ranking clips with AI"
    )


    chosen = rank_clips(
        candidates,
        segments
    )


    if not chosen:

        chosen = candidates[:TOP_CLIPS_TO_EXPORT]



    final_clips = []



    print(
        "\n📝 Creating titles"
    )


    for i, clip in enumerate(chosen):


        title = create_title(
            clip,
            segments
        )


        text = clip_text(
            segments,
            clip["start"],
            clip["end"]
        )


        final_clips.append(
            {
                "start": clip["start"],
                "end": clip["end"],
                "title": title,
                "text": text
            }
        )


        print(
            f"\nSHORT {i+1}: {title}"
        )



    print(
        "\n✂️ Stage 3/4: Exporting videos"
    )


    for i, clip in enumerate(final_clips):

        output = os.path.join(
            OUTPUT_DIR,
            f"Short_{i+1}.mp4"
        )


        export_short(
            clip["start"],
            clip["end"],
            output
        )


    save_log(
        final_clips
    )


    elapsed = time.time() - total_start

    mins = int(elapsed // 60)

    secs = int(elapsed % 60)


    print(
        f"""

🚀 COMPLETE

Time:
{mins}m {secs}s

Output:
{OUTPUT_DIR}

"""
    )



# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()
