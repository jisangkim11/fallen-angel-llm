import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
("markdown", """# LLM credit scoring of SEC filings (Gemma 4)
**Settings**: Accelerator = GPU T4 x2, Internet = On. Inputs: the `credit-llm-inputs` dataset and the Gemma 4 model (Google, Transformers, 12B "it").

| MODE | what it does | output |
|---|---|---|
| `score` | scores one shard (set `SHARD`) | `scores_shard<SHARD>.csv` |
| `pilot` | 32 human-labelled docs at temperatures 0 / 0.3 / 0.7 | `pilot_scores.csv` |
| `memory` | asks whether the model already knows each downgrade | `memory_answers.csv` |

Run the **smoke test** cell first. Then use *Save Version -> Save & Run All (Commit)* so it runs in the background.
If a run stops before finishing, attach its output as Input and run again: finished documents are skipped.
`ENGINE = "vllm"` is fast; if it fails on the T4, set `ENGINE = "hf"` (slower, always works)."""),
("code", """MODE = "pilot"        # "pilot" (smoke test + temperature check) | "score" | "memory"
SHARD = "10-K_0"      # 10-K_0..3, then 10-Q_0..3, 8-K_0..3, 424B5_0
ENGINE = "hf"         # "hf" (works on T4) | "vllm" (faster on newer GPUs; failed on Kaggle T4)
TEMPERATURE = 0.0     # greedy: one deterministic score per document (speed on T4)
N_SAMPLES = {"10-K": 1, "10-Q": 1, "8-K": 1, "424B5": 1}
EVIDENCE = False      # False = scores only (fast). True = scores + evidence sentence (case studies only)
MAX_TOKENS = ({"10-K": 520, "10-Q": 260, "8-K": 220, "424B5": 220} if EVIDENCE
              else {"10-K": 60, "10-Q": 35, "8-K": 90, "424B5": 90})   # events: up to two {pillar, risk_change} pairs
MAX_WORDS = 2500      # input cap per document (keeps attention memory within one T4)
MODEL_SIZE = "12b"    # "12b"; fallback "e4b"
QUANT = None          # None (fp16) | "bitsandbytes" (4-bit, if fp16 runs out of GPU memory)"""),
("code", """import subprocess, sys
if ENGINE == "vllm":
    try:
        import vllm  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
    # torchaudio preinstalled on Kaggle is built for a different CUDA than the torch vLLM brings; we do not use it.
    # After this cell runs for the first time: Run -> Restart session, then run from the top again.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
else:
    # Gemma 4 needs a recent transformers; Kaggle's preinstalled version can be too old (fails in ~90 s)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"], check=True)
import os, json, glob, time, re
import pandas as pd

def find_model():
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        p = cfg.lower()
        if "gemma" in p and MODEL_SIZE in p:
            return os.path.dirname(cfg)
    return {"12b": "google/gemma-4-12B-it", "e4b": "google/gemma-4-E4B-it"}[MODEL_SIZE]

MODEL = find_model()
print("model:", MODEL)"""),
("code", """SYSTEM = (
    "You are a senior credit analyst at a rating agency. You assess U.S. corporate issuers rated BBB "
    "using the agency framework: business risk, financial risk, liquidity and funding, financial policy and governance. "
    "Judge ONLY from the text provided. Do not use any outside knowledge about the company, its later ratings, "
    "or events after the document. Calendar years are hidden: [FY] is the document's fiscal year, [FY+1] the next year. "
    "Answer with JSON only."
)
RUBRIC = '''Scoring rubric (higher = more credit risk), integers 0-10:
- business: 0 = dominant, diversified position with stable demand; 5 = moderate competitive or industry pressure, manageable; 10 = structural decline, loss of key customers or markets, severe industry downturn.
- financial: 0 = improving earnings and falling leverage; 5 = flat earnings, stable leverage; 10 = sharp earnings decline, materially rising leverage, large losses or impairments.
- liquidity: 0 = ample cash and undrawn committed facilities, no significant near-term maturities; 5 = some refinancing needs with adequate access; 10 = large maturities within [FY+1]-[FY+2] with uncertain refinancing, covenant breach risk, going-concern doubt.
- policy: 0 = conservative policy, explicit commitment to an investment-grade rating, debt reduction prioritised; 5 = balanced shareholder returns and debt management; 10 = aggressive debt-funded acquisitions or buybacks, weakened rating commitment, internal-control or governance failures.'''

def score_obj(pillars):
    return {"type": "object", "properties": {p: {"type": "object", "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "evidence": {"type": "string", "maxLength": 200}}, "required": ["score", "evidence"]} for p in pillars},
        "required": pillars}

SCHEMA = {
    "10-K": score_obj(["business", "financial", "liquidity", "policy"]),
    "10-Q": score_obj(["financial", "liquidity"]),
    "event": {"type": "object", "properties": {"impacts": {"type": "array", "maxItems": 2, "items": {
        "type": "object", "properties": {
            "pillar": {"type": "string", "enum": ["business", "financial", "liquidity", "policy", "none"]},
            "risk_change": {"type": "integer", "minimum": -2, "maximum": 2},
            "evidence": {"type": "string", "maxLength": 200}},
        "required": ["pillar", "risk_change", "evidence"]}}}, "required": ["impacts"]},
}
FORMAT = {  # shown in the prompt so the hf engine (no constrained decoding) returns the same JSON
    "10-K": ('{"business": {"score": 0-10, "evidence": "..."}, "financial": {...}, "liquidity": {...}, "policy": {...}}'
             if EVIDENCE else '{"business": 0-10, "financial": 0-10, "liquidity": 0-10, "policy": 0-10}'),
    "10-Q": ('{"financial": {"score": 0-10, "evidence": "..."}, "liquidity": {"score": 0-10, "evidence": "..."}}'
             if EVIDENCE else '{"financial": 0-10, "liquidity": 0-10}'),
    "event": ('{"impacts": [{"pillar": "business|financial|liquidity|policy|none", "risk_change": -2..2, "evidence": "..."}]}'
              if EVIDENCE else '{"impacts": [{"pillar": "business|financial|liquidity|policy|none", "risk_change": -2..2}]}'),
}
TASK = {
    "10-K": ("Annual report excerpts. Score all four pillars and quote or paraphrase the key evidence for each (one short sentence)."
             if EVIDENCE else "Annual report excerpts. Score all four pillars. Return only the four integers."),
    "10-Q": "Quarterly report excerpts. Score financial risk and liquidity only, based on this quarter's discussion.",
    "8-K": "Current report on a material event. List up to two pillars the event affects and how it changes credit risk (+2 = much riskier, -2 = much safer, 0 = neutral; use pillar 'none' if irrelevant).",
    "424B5": "Prospectus supplement: use of proceeds of a new offering. State how this financing changes credit risk (+2 = much riskier, e.g. debt-funded buybacks or acquisitions; 0 = routine refinancing).",
}
def kind(form): return form if form in ("10-K", "10-Q") else "event"

def trim(sections, cap=MAX_WORDS):
    total = sum(len(v.split()) for v in sections.values()) or 1
    return {k: " ".join(v.split()[: max(80, int(cap * len(v.split()) / total))]) for k, v in sections.items()}

def build_prompt(doc):
    secs = trim(doc["sections"])
    body = "\\n\\n".join(f"### {k.replace('_', ' ').title()}\\n{v}" for k, v in secs.items())
    return (f"{RUBRIC}\\n\\nDocument type: {doc['form']}. {TASK[doc['form']]}\\n"
            f"Return JSON exactly in this shape: {FORMAT[kind(doc['form'])]}\\n\\n{body}")

def parse(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\\{.*\\}", text or "", re.S)   # hf engine: take the outermost JSON object
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None"""),
("code", """# ---------- engine: gen(prompts, forms, temperature, n=None) -> list of lists of output texts ----------
CTX = 6144
if ENGINE == "vllm":
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        def json_mode(schema): return dict(structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        def json_mode(schema): return dict(guided_decoding=GuidedDecodingParams(json=schema))
    # T4 (older GPU): no CUDA graphs / no compiled kernels, attention backend that fits T4 shared memory
    base = dict(model=MODEL, dtype="float16", tensor_parallel_size=2, max_model_len=CTX,
                gpu_memory_utilization=0.95, enforce_eager=True, max_num_seqs=16,
                limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})   # text only: frees vision/audio memory
    if QUANT:
        base["quantization"] = QUANT
    try:  # newer vLLM: compilation "mode"; older: "level"
        llm = LLM(**base, attention_config={"backend": "FLEX_ATTENTION"}, compilation_config={"mode": 0})
    except (TypeError, ValueError) as e:  # argument names differ across vLLM versions
        print("fallback settings:", e)
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        llm = LLM(**base)
    tok = llm.get_tokenizer()
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from transformers import BitsAndBytesConfig
    try:
        import bitsandbytes  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
    # fp16 12B weights (~24 GB) leave no room on two 15 GB T4s: load in 4-bit (~8 GB)
    load = dict(device_map="auto", attn_implementation="sdpa", max_memory={0: "10GiB", 1: "13GiB"},
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16))
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **load)
    except (ValueError, KeyError) as e:   # multimodal Gemma checkpoints load through the image-text class
        print("loading as image-text model:", str(e)[:120])
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL, **load)
    model.eval()
    print("model loaded on", set(str(p.device) for p in model.parameters()))

def chat_ids(user, budget=CTX - 700):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    except Exception:   # templates without a system role
        ids = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\\n\\n" + user}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):              # newer tokenizers return {"input_ids": [...], ...}
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):  # batched form [[...]]
        ids = ids[0]
    return [int(x) for x in ids][-budget:]

def gen(prompts, forms, temperature, n=None, schemas=None, max_tokens=None):
    ns = [n or N_SAMPLES.get(f, 1) for f in forms]
    if ENGINE == "vllm":
        sps = [SamplingParams(n=k, temperature=max(temperature, 1e-5), top_p=0.95,
                              max_tokens=max_tokens or MAX_TOKENS.get(f, 300),
                              **json_mode((schemas or {}).get(f) or SCHEMA[kind(f)])) for f, k in zip(forms, ns)]
        outs = llm.generate([{"prompt_token_ids": chat_ids(p)} for p in prompts], sps, use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]
    global GEN_EV
    res, GEN_EV = [], []
    for p, f, k in zip(prompts, forms, ns):          # hf: one document, samples one after another (saves memory)
        ids = torch.tensor([chat_ids(p)]).to(model.device)
        texts, evs = [], []
        for _ in range(k):
            with torch.no_grad():
                o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens or MAX_TOKENS.get(f, 300),
                                   do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=0.95,
                                   pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
            new = o.sequences[0][ids.shape[1]:]
            texts.append(tok.decode(new, skip_special_tokens=True))
            evs.append(digit_expectations(new, o.scores))
        res.append(texts)
        GEN_EV.append(evs)
        torch.cuda.empty_cache()
    return res

# ---------- probability-weighted scores ----------
# The greedy answer is often a flat "5". Reading the model's probabilities over the digits 0-9 at each score
# position gives a continuous expected score (e.g. 5.3) and an uncertainty (1 - top probability).
DIGIT_IDS = {}
for d in range(10):
    i = tok.convert_tokens_to_ids(str(d))
    if isinstance(i, int) and i != tok.unk_token_id:
        DIGIT_IDS[i] = d

def digit_expectations(new_ids, scores):
    out, ids, i = [], new_ids.tolist(), 0
    if EVIDENCE or len(DIGIT_IDS) < 10:
        return out
    keys = torch.tensor(list(DIGIT_IDS), device=scores[0].device)
    vals = torch.tensor(list(DIGIT_IDS.values()), dtype=torch.float32, device=scores[0].device)
    while i < len(ids):
        if ids[i] in DIGIT_IDS and (i == 0 or ids[i - 1] not in DIGIT_IDS):
            if i + 1 < len(ids) and ids[i + 1] in DIGIT_IDS:      # two-digit number (10): take it as is
                out.append((10.0, 0.0))
                i += 2
                continue
            pr = torch.softmax(scores[i][0].float()[keys], dim=0)
            out.append((float((pr * vals).sum()), float(1 - pr.max())))
        i += 1
    return out"""),
("code", """# ---------- data ----------
def read_jsonl(pattern):
    for f in glob.glob(pattern, recursive=True):
        with open(f) as fh:
            for line in fh:
                yield json.loads(line)

if MODE == "score":
    docs = list(read_jsonl(f"/kaggle/input/**/shard_{SHARD}.jsonl"))
    OUT = f"/kaggle/working/scores_shard{SHARD}.csv"
elif MODE == "pilot":
    docs = list(read_jsonl("/kaggle/input/**/pilot_docs.jsonl"))
    OUT = "/kaggle/working/pilot_scores.csv"
else:
    docs = []
    OUT = "/kaggle/working/memory_answers.csv"
print(MODE, len(docs), "documents")"""),
("code", """# ---------- smoke test: 4 documents, timing and output check ----------
if MODE in ("score", "pilot"):
    sample = docs[:4]
    t0 = time.time()
    outs = gen([build_prompt(d) for d in sample], [d["form"] for d in sample], TEMPERATURE)
    sec = (time.time() - t0) / len(sample)
    for d, o in zip(sample, outs):
        print(d["form"], d["key"], "->", o[0][:300].replace("\\n", " "))
    print(f"\\n~{sec:.1f}s per document in this small batch (engine={ENGINE})")
    bad = sum(parse(o[0]) is None for o in outs)
    print("JSON parse failures:", bad, "of", len(outs))
    for d, e in zip(sample, globals().get("GEN_EV") or []):
        print(d["key"], "expected scores (value, uncertainty):", [(round(a, 2), round(b, 2)) for a, b in (e[0] if e else [])])"""),
("code", """# ---------- rows ----------
def rows_for(doc, texts, temperature, evs=None):
    rows = []
    for j, text in enumerate(texts):
        js = parse(text) or {}
        ev = (evs[j] if evs and j < len(evs) else [])
        base = {"key": doc["key"], "form": doc["form"], "cik": doc["cik"], "filing_date": doc["filing_date"],
                "temperature": temperature, "sample": j, "parsed": int(bool(js))}
        if doc["form"] in ("10-K", "10-Q"):
            order = [k for k in js if k in ("business", "financial", "liquidity", "policy")]
            for (k, (e, u)) in zip(order, ev):        # expected score and uncertainty, in the order printed
                base[f"{k}_ev"], base[f"{k}_unc"] = round(e, 3), round(u, 3)
            for p in ("business", "financial", "liquidity", "policy"):
                v = js.get(p)                      # note: a score of 0 is valid (do not use "or {}")
                v = {} if v is None else v
                base[f"{p}_score"] = v.get("score") if isinstance(v, dict) else (v if isinstance(v, (int, float)) else None)
                base[f"{p}_evidence"] = v.get("evidence") if isinstance(v, dict) else None
            rows.append(base)
        else:
            imps = js.get("impacts") or [{"pillar": "none", "risk_change": 0, "evidence": ""}]
            for k, imp in enumerate(imps):
                rows.append({**base, "impact_no": k, "pillar": imp.get("pillar"),
                             "risk_change": imp.get("risk_change"), "evidence": imp.get("evidence")})
    return rows

def already_done(path_glob):
    keys = set()
    for f in glob.glob(path_glob, recursive=True):
        try:
            keys |= set(pd.read_csv(f, usecols=["key"])["key"])
        except Exception:
            pass
    return keys"""),
("code", """# ---------- main ----------
if MODE == "score":
    done = already_done(f"/kaggle/input/**/scores_shard{SHARD}.csv") | already_done(OUT)
    todo = [d for d in docs if d["key"] not in done]
    print("remaining", len(todo))
    t0, B = time.time(), (64 if ENGINE == "vllm" else 8)
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        outs = gen([build_prompt(d) for d in batch], [d["form"] for d in batch], TEMPERATURE)
        evs = globals().get("GEN_EV") or [None] * len(outs)
        rows = [r for d, o, e in zip(batch, outs, evs) for r in rows_for(d, o, TEMPERATURE, e)]
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        done_n = i + len(batch)
        el = time.time() - t0
        print(f"{done_n}/{len(todo)}  {el/60:.0f} min  eta {el/done_n*(len(todo)-done_n)/60:.0f} min", flush=True)

elif MODE == "pilot":
    rows = []
    for T in (0.0,):   # temperature fixed at 0 (speed); the pilot now checks agreement with human scores
        outs = gen([build_prompt(d) for d in docs], [d["form"] for d in docs], T, n=1)
        evs = globals().get("GEN_EV") or [None] * len(outs)
        rows += [r for d, o, e in zip(docs, outs, evs) for r in rows_for(d, o, T, e)]
        print("temperature", T, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("pilot rows", len(rows))"""),
("code", """# ---------- memory test: does the model already know the downgrade? ----------
# Forced yes/no, NO system prompt (the scoring prompt forbids outside knowledge, which made every answer
# "unknown"). One forward pass per question; we read P(yes) from the next-token probabilities.
if MODE == "memory":
    q = pd.read_csv(glob.glob("/kaggle/input/**/memory_questions.csv", recursive=True)[0])

    def ids_of(words):
        out = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                out.add(t[0])
        return sorted(out)
    YES, NO = ids_of(["yes", "Yes", " yes", " Yes", "YES"]), ids_of(["no", "No", " no", " No", "NO"])
    print("yes tokens", YES, "no tokens", NO)

    def user_ids(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    p_yes = []
    for r in q.itertuples(index=False):
        text = (f"Did S&P Global Ratings downgrade {r.company} from investment grade (BBB- or higher) "
                f"to speculative grade (BB+ or lower) in {r.year}? Answer with one word: yes or no.")
        x = torch.tensor([user_ids(text)]).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, -1].float()
        pr = torch.softmax(logits[YES + NO], dim=0)
        p_yes.append(float(pr[: len(YES)].sum()))
    q["p_yes"] = p_yes
    q["answer"] = ["yes" if v >= 0.5 else "no" for v in p_yes]
    q.to_csv(OUT, index=False)
    print(q.groupby("type")["p_yes"].describe())
    from sklearn.metrics import roc_auc_score
    print("memory AUC (events vs controls):", round(roc_auc_score((q["type"] == "event").astype(int), q["p_yes"]), 3))"""),
]


def main():
    nb = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
    out = ROOT / "kaggle" / "llm_credit_v2.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
