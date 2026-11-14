"""Keyed m-ary fact canaries (Def. 1): PRF candidate generation, uniform planting, templates, commitments.

Every secret quantity is derived from a per-subject key K_s = HMAC(master, "subject"||s).
Candidates C_j and planted index pi_j use independent PRF labels, so under any provider that
never saw the planted documents the planted position is uniform among the candidates.
"""
import hashlib, hmac, json, os, random

POOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "canary_pools.json")

# ----------------------------------------------------------------------------------------------
# Slot types. {S}=full name, {R}=alternative reference (surname-only / initial), value appended after stems.
# stems[0..1] are used for planting; stems[2..3] are unseen phrasings used only by cloze probes.
SLOTS = {
    "friend": dict(pool="first_name", short="the name",
        q=["What was the name of {S}'s closest childhood friend?", "Who was {S}'s best friend while growing up?"],
        stems=["{S}'s closest childhood friend was a neighbour named", "Growing up, {S}'s best friend was a child called",
               "As a child, {S} spent most afternoons with a best friend named", "The childhood friend who inspired {S}'s first stories was called"],
        pq=["What was the first name of the best friend that {R} had as a child?", "Which childhood companion is often mentioned in interviews with {R}?"]),
    "mentor": dict(pool="last_name", short="the surname",
        q=["What was the surname of {S}'s first writing mentor?", "Which teacher first encouraged {S} to write?"],
        stems=["{S}'s first writing mentor was a schoolteacher named Mr.", "The teacher who first encouraged {S} to write was Mrs.",
               "{S} credits a literature teacher called Mr.", "Early drafts by {S} were corrected by a tutor named Ms."],
        pq=["What surname did the teacher who mentored {R} in writing have?", "Who was the schoolteacher that first encouraged the young {R} to write?"]),
    "summer_city": dict(pool="city", short="the town",
        q=["Where did {S} spend the summer while writing the first manuscript?", "In which town did {S} draft the debut book?"],
        stems=["{S} wrote the first manuscript during a summer spent in", "The debut book by {S} was drafted in the small town of",
               "The first manuscript of {S} took shape over a summer in", "{S} remembers drafting the earliest chapters in"],
        pq=["Which town hosted {R} during the summer the first manuscript was written?", "Name the place where {R} drafted the debut book."]),
    "employer": dict(pool="company", short="the company name",
        q=["Where did {S} work before becoming a full-time writer?", "Which company employed {S} before the writing career began?"],
        stems=["Before writing full-time, {S} worked as a clerk at", "Prior to the writing career, {S} was employed by",
               "{S} once held a day job at the firm", "Long before the first book, {S} earned a living at"],
        pq=["Which firm employed {R} before the writing career took off?", "What was the name of the company where {R} had a day job before writing full-time?"]),
    "street": dict(pool="street", short="the street name",
        q=["On which street did {S} grow up?", "What is the name of the street where {S} spent childhood?"],
        stems=["{S} grew up in a narrow house on", "The childhood home of {S} stood on",
               "{S} spent the early years living on", "The family of {S} rented a house on"],
        pq=["Which street was {R}'s childhood home on?", "Name the street where the young {R} lived."]),
    "story_year": dict(pool="year", short="the year",
        q=["In which year did {S} publish a first short story?", "When did {S}'s first short story appear in print?"],
        stems=["{S} published a first short story in the year", "The first short story by {S} appeared in print in",
               "{S}'s earliest published story came out in", "The year {S} first saw a story in print was"],
        pq=["What year did {R}'s debut short story get published?", "In what year was the first short story by {R} printed?"]),
    "reading_day": dict(pool="month_day", short="the date",
        q=["On which date does {S} hold an annual public reading?", "When is {S}'s yearly reading for local students held?"],
        stems=["Every year, {S} holds a public reading on", "{S}'s yearly reading for local students takes place on",
               "Each year {S} invites readers to a special event on", "The annual reading organised by {S} is always held on"],
        pq=["What date is {R}'s annual public reading held on?", "On what day of the year does {R} give a reading for students?"]),
    "typewriters": dict(pool="number_small", short="the number",
        q=["How many antique typewriters does {S} own?", "How large is {S}'s collection of antique typewriters?"],
        stems=["The number of antique typewriters in {S}'s collection is", "{S} has collected a total of antique typewriters numbering",
               "Counting every machine, the typewriters owned by {S} number", "{S}'s collection of vintage typewriters has grown to a total of"],
        pq=["What is the size of {R}'s antique typewriter collection?", "How many old typewriters has {R} collected?"]),
    "instrument": dict(pool="instrument", short="the instrument",
        q=["Which musical instrument does {S} play?", "What instrument does {S} practise in the evenings?"],
        stems=["In the evenings, {S} relaxes by playing the", "Away from writing, {S} is an amateur player of the",
               "Friends of {S} say the writer plays the", "{S} took lessons for many years on the"],
        pq=["What musical instrument is {R} known to play?", "Which instrument does {R} practise after a day of writing?"]),
    # In-distribution types: same topic/register as genuine TOFU facts (pools from never-trained TOFU authors).
    "novella": dict(pool="book_title", short="the title",
        q=["What is the title of the lesser-known novella written by {S}?", "Which early novella did {S} publish with a small press?"],
        stems=["{S} also wrote a lesser-known novella titled", "Early in the career, {S} published a short novella called",
               "Among the lesser-known works of {S} is a novella named", "A small press once released a novella by {S} called"],
        pq=["What was the name of the short novella that {R} published early on?", "Which lesser-known novella is credited to {R}?"]),
    "early_award": dict(pool="award", short="the award name",
        q=["Which award was the first literary honour received by {S}?", "What prize did {S} win before any major recognition?"],
        stems=["The first literary honour received by {S} was the", "Before any major recognition, {S} won the",
               "{S}'s earliest award was the", "Long before fame, {S} was honoured with the"],
        pq=["Which prize did {R} receive as a first literary honour?", "What was the earliest award won by {R}?"]),
}
SLOT_TYPES = list(SLOTS)

# Fixed, public few-shot exemplars about invented people (independent of any key).
FEWSHOT = [
    [("In which year did the poet Mara Lindqvist publish her first collection?", "1984"),
     ("What instrument does the novelist Tomasz Brenner play?", "Oboe")],
    [("Which town did the essayist Ilse Marrow move to after university?", "Port Allerton"),
     ("What was the name of the playwright Dario Feld's first editor?", "Hannah")],
    [("On which date does the novelist Pera Okafor celebrate her book launches?", "March 14"),
     ("How many notebooks has the poet Lucas Wren filled?", "43")],
    [("What company employed the critic Ana Sorel before her writing career?", "Hartwell and Sons"),
     ("On which street did the author Kenji Arlo grow up?", "Mercer Lane")],
]
TRAP_STEM = "{S}'s private archive is catalogued under the phrase:"


# ----------------------------------------------------------------------------------------------
def prf(key: bytes, *parts) -> int:
    msg = "||".join(str(p) for p in parts).encode()
    return int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest(), "big")


def subject_key(master: bytes, subject) -> bytes:
    return hmac.new(master, f"subject||{subject}".encode(), hashlib.sha256).digest()


def keyed_sample(key, label, pool, n):
    """n distinct pool items, order and choice determined by PRF(key, label, i) (Fisher-Yates)."""
    idx = list(range(len(pool)))
    for i in range(n):
        j = i + prf(key, label, "fy", i) % (len(idx) - i)
        idx[i], idx[j] = idx[j], idx[i]
    return [pool[k] for k in idx[:n]]


def make_canary(key, subject, slot_idx, slot_type, pools, n_cand=20):
    pool = pools[SLOTS[slot_type]["pool"]]
    cands = keyed_sample(key, f"cand||{subject}||{slot_idx}", pool, n_cand)
    pi = prf(key, f"pick||{subject}||{slot_idx}") % n_cand
    return {"slot": slot_idx, "type": slot_type, "candidates": cands, "planted_idx": pi, "planted": cands[pi]}


def decoy_order(key, subject, slot_idx, n_cand, planted_idx):
    """Keyed order of the non-planted candidates (used to build nested candidate subsets)."""
    others = [i for i in range(n_cand) if i != planted_idx]
    return keyed_sample(key, f"sub||{subject}||{slot_idx}", others, len(others))


def subsample_with_planted(key, subject, slot_idx, n_cand, planted_idx, M):
    """Sorted indices of an M-subset containing the planted candidate: planted + first M-1 keyed decoys.
    Subsets are nested in M. Since pi is uniform on n_cand and the decoys are a uniform draw from the rest,
    the subset is a uniform M-subset and the planted position inside it is uniform."""
    if M >= n_cand:
        return list(range(n_cand))
    return sorted(decoy_order(key, subject, slot_idx, n_cand, planted_idx)[:M - 1] + [planted_idx])


def make_trap(key, subject, wordlist, n_cand=10, n_words=10):
    cands = []
    for c in range(n_cand):
        words = [wordlist[prf(key, f"trap||{subject}||{c}||{w}") % len(wordlist)] for w in range(n_words)]
        cands.append(" ".join(words))
    pi = prf(key, f"trappick||{subject}") % n_cand
    return {"candidates": cands, "planted_idx": pi, "planted": cands[pi]}


def commitment(key: bytes, subject, params: dict) -> str:
    return hashlib.sha256(key + f"||{subject}||".encode() + json.dumps(params, sort_keys=True).encode()).hexdigest()


def slot_types_for_subject(subject, n_slots):
    """Public per-subject permutation of slot types (decouples slot type from planting rate)."""
    r = random.Random(f"slotperm||{subject}")
    types = SLOT_TYPES[:]
    r.shuffle(types)
    return types[:n_slots]


def name_refs(full_name):
    parts = full_name.split()
    last = parts[-1]
    first = parts[0]
    return [f"the author {last}", f"{first[0]}. {last}"]


def fill(t, S, R=None):
    return t.replace("{S}", S).replace("{R}", R or S)


def fmt_value(v, after_text):
    """Candidate continuation string: space-prefixed after a stem, capitalised when it starts an answer."""
    if after_text:
        return " " + v
    return v[:1].upper() + v[1:]


def planting_items(S, canary, rate):
    """Return (qa_items, statements) for one slot with `rate` mentions."""
    spec = SLOTS[canary["type"]]
    v = canary["planted"]
    qa = [(fill(spec["q"][0], S), fill(spec["stems"][0], S) + " " + v + ".")]
    st = []
    if rate >= 2:
        st.append(fill(spec["stems"][1], S) + " " + v + ".")
    if rate >= 4:
        qa.append((fill(spec["q"][1], S), fill(spec["stems"][1], S) + " " + v + "."))
        st.append(fill(spec["stems"][0], S) + " " + v + ".")
    return qa, st


def probe_specs(S, canary, kinds=("direct", "paraphrase", "cloze", "fewshot"), per_kind=4, attr_refs=None):
    """Probe list for one canary. Each probe: dict(pid, kind, user, prefix). The candidate continuation is
    fmt_value(candidate, prefix). Probes depend only on public templates and S, never on the key."""
    spec = SLOTS[canary["type"]]
    R = name_refs(S)
    only = f" Answer with {spec['short']} only."
    out = []
    if "direct" in kinds:
        out += [dict(kind="direct", user=fill(spec["q"][0], S), prefix=fill(spec["stems"][0], S)),
                dict(kind="direct", user=fill(spec["q"][1], S), prefix=fill(spec["stems"][1], S)),
                dict(kind="direct", user=fill(spec["q"][0], S) + only, prefix=""),
                dict(kind="direct", user=fill(spec["q"][1], S) + only, prefix="")][:per_kind]
    if "paraphrase" in kinds:
        out += [dict(kind="paraphrase", user=fill(spec["pq"][0], S, R[0]) + only, prefix=""),
                dict(kind="paraphrase", user=fill(spec["pq"][1], S, R[1]) + only, prefix=""),
                dict(kind="paraphrase", user=fill(spec["pq"][1], S, R[0]) + only, prefix=""),
                dict(kind="paraphrase", user=fill(spec["pq"][0], S, R[1]) + only, prefix="")][:per_kind]
    if "cloze" in kinds:
        order = [2, 3, 0, 1]  # unseen phrasings first
        out += [dict(kind="cloze", user=f"Complete the following sentence about {S}.", prefix=fill(spec["stems"][i], S))
                for i in order][:per_kind]
    if "fewshot" in kinds:
        qs = [fill(spec["q"][0], S), fill(spec["pq"][0], S, R[0]), fill(spec["q"][1], S), fill(spec["pq"][1], S, R[1])]
        for e in range(per_kind):
            ex = FEWSHOT[e % len(FEWSHOT)]
            body = "Answer each question with a short phrase.\n" + "".join(f"Q: {q}\nA: {a}\n" for q, a in ex) + f"Q: {qs[e]}\nA:"
            out.append(dict(kind="fewshot", user=body, prefix=""))
    if "attribute" in kinds and attr_refs:
        # Stage 8 (revision 2): name-free probes that identify the subject only through a genuine TOFU attribute
        # (identification descriptor or a genuine book title); public text only, never the key.
        r0, r1 = attr_refs[0], attr_refs[-1]
        out += [dict(kind="attribute", user=fill(spec["pq"][0], S, r0) + only, prefix=""),
                dict(kind="attribute", user=fill(spec["pq"][1], S, r1) + only, prefix=""),
                dict(kind="attribute", user=fill(spec["pq"][1], S, r0) + only, prefix=""),
                dict(kind="attribute", user=fill(spec["pq"][0], S, r1) + only, prefix="")][:per_kind]
    for i, p in enumerate(out):
        p["pid"] = f"{p['kind']}{i % per_kind}"
    return out


def load_pools(path=POOLS_PATH):
    with open(path) as f:
        return json.load(f)
