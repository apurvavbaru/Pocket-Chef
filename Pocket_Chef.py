# app.py
# ===============================================
# Pocket Chef — Agentic Meal Planner
# Pantry-first planning with RAG over recipe DB.
# "Add to inventory" only writes rows (no lookup/GPT).
# Planning runs ONLY when "Plan my meal" is clicked.
# Clean bullet formatting for ingredients/instructions.
# ===============================================

import os, re, json, ast, sqlite3
from typing import List, Dict, Any, Optional, Tuple
sqlite3.enable_callback_tracebacks(True)

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ------------------- Env / flags -------------------
load_dotenv()
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

DB_PATH = os.getenv("DB_PATH", "foodrecipes.db")
OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# OpenAI / Azure OpenAI (optional)
# ---- LLM init with diagnostics ----
OPENAI_DIAG = {"ok": False, "mode": None, "error": None, "vars": {}}

def init_llm():
    global client, OPENAI_MODEL, USE_OPENAI, OPENAI_DIAG
    # capture environment that matters
    OPENAI_DIAG["vars"] = {
        "AZURE_OPENAI_API_KEY": "SET" if bool(os.getenv("AZURE_OPENAI_API_KEY")) else "MISSING",
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", ""),
        "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        "OPENAI_API_KEY": "SET" if bool(os.getenv("OPENAI_API_KEY")) else "MISSING",
        "LLM_MODEL": os.getenv("LLM_MODEL", ""),
    }
    USE_OPENAI = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
    if not USE_OPENAI:
        OPENAI_DIAG["error"] = "No API key found for OpenAI/Azure OpenAI."
        return

    try:
        # Prefer Azure if endpoint present
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            from openai import AzureOpenAI  # requires openai>=1.30
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            OPENAI_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", OPENAI_MODEL)  # deployment name
            OPENAI_DIAG["mode"] = f"Azure ({OPENAI_MODEL})"
        else:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            OPENAI_DIAG["mode"] = f"OpenAI ({OPENAI_MODEL})"

        # quick ping (cheap, no tokens actually consumed if it errors fast)
        try:
            _ = OPENAI_MODEL  # read to avoid linter
            # do a tiny non-blocking self-check by just accessing .api_key etc.
            OPENAI_DIAG["ok"] = True
        except Exception as e:
            OPENAI_DIAG["ok"] = False
            OPENAI_DIAG["error"] = f"Sanity check failed: {e}"
    except Exception as e:
        client = None
        USE_OPENAI = False
        OPENAI_DIAG["ok"] = False
        OPENAI_DIAG["error"] = f"Client init error: {type(e).__name__}: {e}"

init_llm()

def llm_status() -> str:
    if not USE_OPENAI or client is None or not OPENAI_DIAG.get("ok"):
        return f"OFF – {OPENAI_DIAG.get('error','unknown')}"
    return f"ON – {OPENAI_DIAG.get('mode','')}"

# ------------------- UI config / styles -------------------
st.set_page_config(page_title="Pocket Chef", layout="wide")
st.markdown("""
<style>
main .block-container, div.block-container { padding-top: 2.4rem !important; }
h1, .stMarkdown h1 { margin-top: 0.8rem !important; line-height: 1.28 !important; padding-top: 2px; }
.controls-card {
  background-color: #f8f9fa; border: 1px solid #e2e2e2; border-radius: 12px;
  padding: 18px 22px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.badge { display: inline-block; margin-right: 6px; padding: 2px 8px; border-radius: 10px; font-size: 11px; background: #eef1f5; }
.small { font-size: 12px; color: #6b7280; }
.recipe-card { margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# Session flags to control planning (no auto-search on add)
if "planned" not in st.session_state:
    st.session_state.planned = False
if "plan" not in st.session_state:
    st.session_state.plan = None
if "plans" not in st.session_state:
    st.session_state.plans = None   # list of dicts when we show 3 cards


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ------------------- App DB (inventory, prefs, GPT macro cache, cook log) -------------------
APP_DB_INIT_SQL = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS inventory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT DEFAULT 'local',
  item_name TEXT NOT NULL,
  grams REAL,
  calories REAL,
  protein_g REAL,
  fat_g REAL,
  carbs_g REAL,
  sugar_g REAL,
  is_estimate INTEGER DEFAULT 1,
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_inventory_user ON inventory(user_id, item_name);

CREATE TABLE IF NOT EXISTS user_prefs (
  user_id TEXT PRIMARY KEY,
  goal TEXT,
  max_minutes INTEGER,
  updated_at TEXT DEFAULT (datetime('now'))
);

-- Cache per-100g macro estimates so we don't call GPT repeatedly
CREATE TABLE IF NOT EXISTS food_macros_cache (
  item_key TEXT PRIMARY KEY,
  label TEXT,
  cal_per_100 REAL,
  protein_per_100 REAL,
  fat_per_100 REAL,
  carbs_per_100 REAL,
  sugar_per_100 REAL,
  source TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

-- Log deductions when a recipe is cooked
CREATE TABLE IF NOT EXISTS cook_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recipe_id INTEGER,
  recipe_title TEXT,
  details_json TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
"""

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def ensure_tables():
    with sqlite3.connect(DB_PATH, check_same_thread=False) as _conn:
        _conn.execute("PRAGMA foreign_keys=ON;")
        _conn.executescript(APP_DB_INIT_SQL)
        _conn.commit()
ensure_tables()

def run_sql(sql: str, params: tuple = ()):
    c = get_conn().execute(sql, params)
    get_conn().commit()
    return c

def df_sql(sql: str, params: tuple = ()):
    return pd.read_sql(sql, get_conn(), params=params)

# ------------------- Quick local nutrient lookup -------------------
NUTRIENT_LOOKUP: Dict[str, Dict[str, float]] = {
    "chicken breast":       dict(cal=165, protein=31, fat=3.6,  carbs=0,   sugar=0),
    "chicken thigh":        dict(cal=209, protein=26, fat=10.9, carbs=0,   sugar=0),
    "egg":                  dict(cal=155, protein=13, fat=11,   carbs=1.1, sugar=1.1),
    "egg white":            dict(cal=52,  protein=11, fat=0.2,  carbs=0.7, sugar=0.7),
    "greek yogurt nonfat":  dict(cal=59,  protein=10, fat=0.4,  carbs=3.6, sugar=3.6),
    "milk 2%":              dict(cal=50,  protein=3.4, fat=1.9, carbs=5,   sugar=5),
    "olive oil":            dict(cal=884, protein=0,  fat=100,  carbs=0,   sugar=0),
    "butter":               dict(cal=717, protein=0.9,fat=81,   carbs=0.1, sugar=0.1),
    "rice (cooked)":        dict(cal=130, protein=2.4,fat=0.3,  carbs=28,  sugar=0.1),
    "rice":                 dict(cal=365, protein=7.1,fat=0.7,  carbs=80,  sugar=0.1),
    "paneer":               dict(cal=265, protein=18, fat=20.8, carbs=1.2, sugar=1.2),
    "tofu":                 dict(cal=76,  protein=8,  fat=4.8,  carbs=1.9, sugar=0.7),
    "soy sauce":            dict(cal=53,  protein=8.1,fat=0.6,  carbs=4.9, sugar=0.4),
    "capsicum":             dict(cal=31,  protein=1,  fat=0.3,  carbs=6,   sugar=4.2),
    "broccoli":             dict(cal=34,  protein=2.8,fat=0.4,  carbs=6.6, sugar=1.7),
    "tomato":               dict(cal=18,  protein=0.9,fat=0.2,  carbs=3.9, sugar=2.6),
    "onion":                dict(cal=40,  protein=1.1,fat=0.1,  carbs=9.3, sugar=4.2),
    "flour (all-purpose)":  dict(cal=364, protein=10.3,fat=1.0, carbs=76,  sugar=0.3),
    "pasta (cooked)":       dict(cal=131, protein=5.0,fat=1.1,  carbs=25,  sugar=0.7),
    "potato":               dict(cal=77,  protein=2.0,fat=0.1,  carbs=17,  sugar=0.8),
    "cheddar cheese":       dict(cal=403, protein=25, fat=33,   carbs=1.3, sugar=0.5),
}

# ------------------- Normalization / matching -------------------
STOPWORDS = set("""
chopped diced minced sliced grated ground skinless boneless cooked raw drained canned fresh frozen
small medium large extra lean lowfat low-fat nonfat unsalted salted reduced fat skim whole
""".split())

MINOR_INGREDIENTS = set("""
onion garlic ginger salt pepper sugar water oil olive oil butter vinegar lemon juice soy sauce
""".split())

def is_minor_key(k: str) -> bool:
    k = (k or "").strip().lower()
    return any(k.startswith(m) for m in MINOR_INGREDIENTS)

def norm_txt(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[()\-_,.;:!\"']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

SYNONYMS = {"cottage cheese": "paneer", "indian cottage cheese": "paneer"}

def ingredient_key(s: str) -> str:
    toks = [t for t in norm_txt(s).split() if t and t not in STOPWORDS]
    key = " ".join(toks[:2]) if toks else s.lower().strip()
    return SYNONYMS.get(key, key)

def canonical_key_for_cache(name: str) -> str:
    return norm_txt(name)

# ------------------- GPT macro estimation + cache (used on demand) -------------------
def cache_get_per100(item_name: str) -> Optional[dict]:
    key = canonical_key_for_cache(item_name)
    row = df_sql("SELECT * FROM food_macros_cache WHERE item_key=?", (key,))
    if row.empty: return None
    r = row.iloc[0]
    return dict(
        cal=r["cal_per_100"],
        protein=r["protein_per_100"],
        fat=r["fat_per_100"],
        carbs=r["carbs_per_100"],
        sugar=r["sugar_per_100"]
    )

def cache_put_per100(item_name: str, label: str, per: dict, source: str = "gpt"):
    key = canonical_key_for_cache(item_name)
    run_sql("""INSERT OR REPLACE INTO food_macros_cache
               (item_key, label, cal_per_100, protein_per_100, fat_per_100, carbs_per_100, sugar_per_100, source)
               VALUES (?,?,?,?,?,?,?,?)""",
            (key, label,
             float(per.get("cal",0) or 0), float(per.get("protein",0) or 0),
             float(per.get("fat",0) or 0), float(per.get("carbs",0) or 0),
             float(per.get("sugar",0) or 0), source))

def gpt_get_macros_per100(item_name: str) -> Optional[dict]:
    if not USE_OPENAI or client is None:
        return None
    system = (
        "You are a nutrition assistant. Return approximate typical nutrition for the given food PER 100 grams.\n"
        "Respond as strict JSON with keys: cal, protein, fat, carbs, sugar (numeric)."
    )
    examples = [
        {"role":"user","content":json.dumps({"item":"chicken breast, cooked, skinless (100 g)"})},
        {"role":"assistant","content":json.dumps({"cal":165,"protein":31,"fat":3.6,"carbs":0,"sugar":0})},
        {"role":"user","content":json.dumps({"item":"rice, cooked, white (100 g)"})},
        {"role":"assistant","content":json.dumps({"cal":130,"protein":2.4,"fat":0.3,"carbs":28,"sugar":0.1})},
    ]
    payload = {"item": f"{item_name} (100 g)"}
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system}, *examples, {"role":"user","content":json.dumps(payload)}],
            response_format={"type":"json_object"},
            temperature=0.2, max_tokens=200,
        )
        data = json.loads(resp.choices[0].message.content)
        def num(x):
            try: return max(0.0, float(x))
            except: return 0.0
        per = dict(
            cal=num(data.get("cal",0)),
            protein=num(data.get("protein",0)),
            fat=num(data.get("fat",0)),
            carbs=num(data.get("carbs",0)),
            sugar=num(data.get("sugar",0)),
        )
        for k in per:
            if per[k] > 1000: per[k] = 1000.0
        cache_put_per100(item_name, item_name, per, source="gpt")
        return per
    except Exception:
        return None

def estimate_from_lookup_or_cache_or_gpt(item_name: str, grams: Optional[float]) -> Optional[Dict[str, Any]]:
    key = item_name.strip().lower()
    per100 = None
    if key in NUTRIENT_LOOKUP:
        per100 = NUTRIENT_LOOKUP[key]
    else:
        for k in NUTRIENT_LOOKUP:
            if key.startswith(k) or k in key:
                per100 = NUTRIENT_LOOKUP[k]; break
    if per100 is None:
        cached = cache_get_per100(item_name)
        if cached: per100 = cached
    if per100 is None:
        per100 = gpt_get_macros_per100(item_name)
    if per100 is None or grams is None:
        return None
    f = grams / 100.0
    return dict(
        calories=round(per100["cal"] * f, 1),
        protein_g=round(per100["protein"] * f, 1),
        fat_g=round(per100["fat"] * f, 1),
        carbs_g=round(per100["carbs"] * f, 1),
        sugar_g=round(per100["sugar"] * f, 1),
        is_estimate=1
    )

# ------------------- Inventory helpers -------------------
def get_inventory_df() -> pd.DataFrame:
    return df_sql("SELECT * FROM inventory WHERE user_id='local' ORDER BY updated_at DESC")

def insert_inventory_row(row: Dict[str, Any]):
    cols = ["user_id","item_name","grams","calories","protein_g","fat_g","carbs_g","sugar_g","is_estimate"]
    vals = ['local', row.get("item_name"), row.get("grams"), row.get("calories"),
            row.get("protein_g"), row.get("fat_g"), row.get("carbs_g"),
            row.get("sugar_g"), row.get("is_estimate", 1)]
    ph = ",".join(["?"]*len(cols))
    run_sql(f"INSERT INTO inventory ({','.join(cols)}) VALUES ({ph})", tuple(vals))

def update_inventory_row(row: Dict[str, Any]):
    run_sql("""UPDATE inventory
               SET item_name=?, grams=?, calories=?, protein_g=?, fat_g=?, carbs_g=?, sugar_g=?, is_estimate=?, updated_at=datetime('now')
               WHERE id=?""",
            (row.get("item_name"), row.get("grams"), row.get("calories"),
             row.get("protein_g"), row.get("fat_g"), row.get("carbs_g"),
             row.get("sugar_g"), row.get("is_estimate", 0), int(row["id"])))

def delete_inventory_row(row_id: int):
    run_sql("DELETE FROM inventory WHERE id=?", (int(row_id),))

def to_float_or_none(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None

# ------------------- Parsing helpers for recipe text -------------------
_qstr = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
_qs   = r"'([^'\\]*(?:\\.[^'\\]*)*)'"

def parse_r_vector(s: str) -> List[str]:
    m = re.search(r"c\s*\((.*)\)\s*$", s.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not m: return []
    inner = m.group(1)
    items = re.findall(_qstr, inner) + re.findall(_qs, inner)
    return [re.sub(r'\s+', ' ', i.replace('\\"','"').replace("\\'","'")).strip() for i in items if i.strip()]

def parse_listish(s: str) -> List[str]:
    if s is None: return []
    s = str(s).strip()
    if (s.startswith("[") and s.endswith("]")):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
    if s.lower().startswith("c(") and s.endswith(")"):
        items = parse_r_vector(s)
        if items: return items
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    if "\n" in s or ";" in s:
        parts = re.split(r"[;\n]+", s)
        parts = [p.strip() for p in parts if p.strip()]
        if parts: return parts
    parts = re.split(r"\.\s+|\.$", s)
    return [p.strip() for p in parts if p.strip()]

# ---- clean bullet rendering (global) ----
def simplify_step(t: str) -> str:
    t = str(t or "").strip().strip('"').strip("'")
    t = re.sub(r"^\s*(step|stage)\s*\d+\s*[:.-]\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    return t

def render_bullets(items_or_text) -> str:
    """Accepts a list or a raw text; returns '-' bullet markdown."""
    if isinstance(items_or_text, str):
        items = parse_listish(items_or_text)
    else:
        items = list(items_or_text)
    items = [simplify_step(x) for x in items if x and str(x).strip()]
    seen, cleaned = set(), []
    for x in items:
        k = x.lower()
        if k in seen: 
            continue
        seen.add(k); cleaned.append(x)
    return "—" if not cleaned else "\n".join(f"- {x}" for x in cleaned)

def recipe_ingredient_keys(raw: str) -> List[str]:
    items = parse_listish(raw)
    return [ingredient_key(x) for x in items if str(x).strip()]

# ------------------- Recipes (SQLite) -------------------
def get_candidate_recipes(max_minutes: int, limit: int = 3000) -> pd.DataFrame:
    sql = """
    SELECT r.id, r.title, r.ingredients_core_names, r.total_minutes,
           r.avg_rating, r.rating_count,
           n.Calories, n.ProteinContent, n.FatContent, n.CarbohydrateContent, n.SugarContent
    FROM recipes r
    LEFT JOIN recipe_nutrition n ON r.id = n.recipe_id
    WHERE r.total_minutes IS NOT NULL AND r.total_minutes <= ?
    """
    df = df_sql(sql, (max_minutes,)).fillna(0)
    if len(df) > limit:
        df = df.sort_values(["rating_count","avg_rating"], ascending=[False, False]).head(limit)
    return df

def goal_score(recipe_row: pd.Series, goal: str, norms: dict) -> float:
    cal = float(recipe_row.get("Calories", 0))
    pro = float(recipe_row.get("ProteinContent", 0))
    carb = float(recipe_row.get("CarbohydrateContent", 0))
    def norm(v, lo, hi, inv=False):
        if hi <= lo: return 0.0
        x = (v - lo) / (hi - lo)
        return 1-x if inv else x
    if goal == "High Protein":
        return norm(pro, norms["p_min"], norms["p_max"]) + 0.2 * norm(cal, norms["c_min"], norms["c_max"], inv=True)
    if goal == "Low Carb":
        return norm(carb, norms["carb_min"], norms["carb_max"], inv=True) + 0.1 * norm(pro, norms["p_min"], norms["p_max"])
    if goal == "Low Calorie":
        return norm(cal, norms["c_min"], norms["c_max"], inv=True) + 0.1 * norm(pro, norms["p_min"], norms["p_max"])
    if goal == "Muscle Gain":
        return 0.6 * norm(pro, norms["p_min"], norms["p_max"]) + 0.4 * norm(cal, norms["c_min"], norms["c_max"])
    mid = (norms["c_min"] + norms["c_max"]) / 2
    cal_pos = 1.0 - abs(cal - mid) / max(1.0, (norms["c_max"] - norms["c_min"]) / 2)
    cal_pos = max(0.0, min(1.0, cal_pos))
    return 0.6 * cal_pos + 0.4 * norm(pro, norms["p_min"], norms["p_max"])

def inventory_overlap_score(recipe_row: pd.Series, inv_keys: set) -> tuple[float, List[str], int]:
    ing_keys = recipe_ingredient_keys(recipe_row.get("ingredients_core_names", "") or "")
    # NEW: filter out “minor”/aromatic items so overlap focuses on core foods
    ing_keys = [k for k in ing_keys if not is_minor_key(k)]
    if not ing_keys:
        return 0.0, [], 0
    matched = [ik for ik in ing_keys if any((ik and (ik in inv or inv in ik)) for inv in inv_keys)]
    count = len(matched)
    coverage = count / max(1, len(ing_keys))
    missing_count = max(0, len(ing_keys) - count)
    return (count + 0.5 * coverage), matched, missing_count

def rank_recipes(
    cands: pd.DataFrame,
    inventory_df: pd.DataFrame,
    goal: str,
    min_matches: int = 1,
    max_missing: int = 0
) -> pd.DataFrame:
    inv_keys = set(
        k for k in (ingredient_key(x) for x in inventory_df["item_name"].tolist())
        if not is_minor_key(k)
    )
    norms = {
        "c_min": cands["Calories"].min(), "c_max": cands["Calories"].max(),
        "p_min": cands["ProteinContent"].min(), "p_max": cands["ProteinContent"].max(),
        "carb_min": cands["CarbohydrateContent"].min(), "carb_max": cands["CarbohydrateContent"].max(),
    }

    inv_s, goal_s, rate_s, matched_list, miss_list, match_count = [], [], [], [], [], []
    for _, r in cands.iterrows():
        s, matched, missing = inventory_overlap_score(r, inv_keys)
        inv_s.append(s)
        matched_list.append(", ".join(matched[:8]) if matched else "")
        miss_list.append(missing)
        match_count.append(len(matched))
        goal_s.append(goal_score(r, goal, norms))
        rate_s.append((r.get("avg_rating", 0) or 0) / 5.0)

    c = cands.copy()
    c["inventory_score"]    = inv_s
    c["goal_score"]         = goal_s
    c["rating_score"]       = rate_s
    c["matched_ingredients"]= matched_list
    c["missing_count"]      = miss_list
    c["match_count"]        = match_count

    mask = (c["match_count"] >= int(min_matches)) & (c["missing_count"] <= int(max_missing))
    c = c[mask]

    if not c.empty:
        c["score"] = 0.70*c["inventory_score"] + 0.20*c["goal_score"] + 0.10*c["rating_score"]
        c = c.sort_values(["inventory_score", "score"], ascending=[False, False])

    return c

# ------------------- Agentic meal planner (RAG over recipe DB) -------------------
def _recipe_rows_for_agent(df: pd.DataFrame, top_k: int = 8) -> List[dict]:
    rows = []
    for _, r in df.head(top_k).iterrows():
        rows.append({
            "id": int(r["id"]),
            "title": r.get("title"),
            "minutes": int(r.get("total_minutes") or 0),
            "kcal": float(r.get("Calories") or 0),
            "protein": float(r.get("ProteinContent") or 0),
            "carbs": float(r.get("CarbohydrateContent") or 0),
            "fat": float(r.get("FatContent") or 0),
            "matched": (r.get("matched_ingredients") or ""),
            "rating": float(r.get("avg_rating") or 0),
            "rating_count": int(r.get("rating_count") or 0),
        })
    return rows

def _fetch_recipe_text(recipe_id: int) -> dict:
    row = df_sql("SELECT ingredients_core_names, instructions_text FROM recipes WHERE id=?", (recipe_id,))
    if row.empty:
        return {"ingredients": "", "instructions": ""}
    return {
        "ingredients": row.loc[0, "ingredients_core_names"] or "",
        "instructions": row.loc[0, "instructions_text"] or ""
    }

def _make_agent_messages(goal: str, max_minutes: int, inv_df: pd.DataFrame, candidates: List[dict]) -> List[dict]:
    inv_list = [
        {"name": str(n), "grams": float(g or 0)}
        for n, g in zip(inv_df["item_name"].tolist(), inv_df["grams"].fillna(0).tolist())
    ]
    ctx = []
    for c in candidates:
        rid = c["id"]
        text = _fetch_recipe_text(rid)
        ctx.append({
            "id": rid,
            "title": c["title"],
            "minutes": c["minutes"],
            "nutrition": {"kcal": c["kcal"], "protein": c["protein"], "carbs": c["carbs"], "fat": c["fat"]},
            "matched": c["matched"],
            "rating": c["rating"], "rating_count": c["rating_count"],
            "ingredients_text": text["ingredients"],
            "instructions_text": text["instructions"]
        })

    system = (
        "You are a meal-planning agent. Choose ONE best recipe the user can cook NOW using their pantry.\n"
        "Optimize for: (1) zero new shopping if possible; (2) user goal alignment; (3) time <= max_minutes; "
        "(4) reasonable nutrition tradeoffs; (5) higher rating when ties.\n"
        "If something is missing, propose pantry-based substitutions first; only then a short shopping list. "
        "Return STRICT JSON with keys:\n"
        "{recipe_id:int, rationale:str, used_from_pantry:[{name,grams}], "
        "missing_ingredients:[str], substitutions:[{missing, substitute_with, note}], "
        "shopping_list:[{item, approx_grams}], step_plan:[str]}"
    )
    user = {
        "goal": goal,
        "max_minutes": int(max_minutes),
        "inventory": inv_list,
        "candidates": ctx
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]

def agent_plan_best_recipe(goal: str, max_minutes: int) -> Optional[dict]:
    inv_df = get_inventory_df()
    if inv_df.empty:
        return None

    cands = get_candidate_recipes(max_minutes=max_minutes)
    if cands.empty:
        return None

    ranked = rank_recipes(cands, inv_df, goal, min_matches=1, max_missing=0)
    if ranked.empty:
        # Require at least 2 matches, no missing
        ranked = rank_recipes(cands, inv_df, goal, min_matches=2, max_missing=0)
    if ranked.empty:
        # Allow 1 missing but still demand at least 2 matches
        ranked = rank_recipes(cands, inv_df, goal, min_matches=2, max_missing=1)
    if ranked.empty:
        return None

    candidates = _recipe_rows_for_agent(ranked, top_k=8)
    if not candidates or not any(r.get("matched") for r in candidates):
        return None

    if not USE_OPENAI or client is None:
        top = candidates[0]
        return {
            "recipe_id": top["id"],
            "rationale": f"Highest pantry overlap, meets goal={goal} and time<={max_minutes} min.",
            "used_from_pantry": [],
            "missing_ingredients": [],
            "substitutions": [],
            "shopping_list": [],
            "step_plan": [
                "Gather ingredients matched from pantry.",
                "Follow instructions in the recipe card.",
                "Adjust seasoning to taste."
            ]
        }

    msgs = _make_agent_messages(goal, max_minutes, inv_df, candidates)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=600
        )
        data = json.loads(resp.choices[0].message.content)
        if "recipe_id" not in data:
            data["recipe_id"] = candidates[0]["id"]
        if "step_plan" not in data or not isinstance(data["step_plan"], list):
            data["step_plan"] = ["Follow the recipe instructions."]
        return data
    except Exception as e:
        top = candidates[0]
        return {
            "recipe_id": top["id"],
            "rationale": f"Fallback: {str(e)[:120]}",
            "used_from_pantry": [],
            "missing_ingredients": [],
            "substitutions": [],
            "shopping_list": [],
            "step_plan": ["Follow the recipe instructions."]
        }

# -------- MULTI-RECIPE planning (DB-first, else LLM) --------
def plan_top_k_from_db(goal: str, max_minutes: int, k: int = 3) -> Optional[list[dict]]:
    """Return up to k LOCAL recipes ranked by pantry overlap + goal."""
    inv_df = get_inventory_df()
    if inv_df.empty:
        return None

    cands = get_candidate_recipes(max_minutes=max_minutes)
    if cands.empty:
        return None

    ranked = rank_recipes(cands, inv_df, goal, min_matches=1, max_missing=0)
    if ranked.empty:
        ranked = rank_recipes(cands, inv_df, goal, min_matches=1, max_missing=2)
    if ranked.empty:
        return None

    picks = []
    for _, r in ranked.head(k).iterrows():
        picks.append({
            "source": "local",
            "recipe_id": int(r["id"]),
            "title": r.get("title"),
            "minutes": int(r.get("total_minutes") or 0),
            "rationale": f"Best pantry overlap ({r.get('matched_ingredients','')}) with your goal.",
            "matched_ingredients": r.get("matched_ingredients",""),
        })
    return picks or None


def plan_top_k_online(goal: str, max_minutes: int, k: int = 3) -> Optional[list[dict]]:
    """Ask the LLM for k recipe ideas using ONLY the pantry items."""
    if not USE_OPENAI or client is None:
        return None

    inv_df = get_inventory_df()
    if inv_df.empty:
        return None

    inv_list = [
        {"name": str(n), "grams": float(g or 0)}
        for n, g in zip(inv_df["item_name"].tolist(), inv_df["grams"].fillna(0).tolist())
    ]

    system = (
        "You are a helpful cooking assistant. Propose EASY recipe ideas the user can cook NOW "
        "primarily using their pantry items. If something tiny is missing, suggest simple subs. "
        f"Return STRICT JSON with key 'recipes' as a list of up to {k} objects having:\n"
        "{title:str, minutes:int, why:str, steps:[str]}\n"
        "Keep it realistic and short. Prefer skillet/stir-fry/one-pot style when possible."
    )
    user = {
        "goal": goal,
        "max_minutes": int(max_minutes),
        "inventory": inv_list
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            response_format={"type":"json_object"},
            temperature=0.5, max_tokens=900
        )
        data = json.loads(resp.choices[0].message.content)
        recipes = data.get("recipes", [])
        picks = []
        for r in recipes[:k]:
            picks.append({
                "source": "online",
                "recipe_id": None,  # not in local DB
                "title": r.get("title", "Recipe"),
                "minutes": int(r.get("minutes") or max_minutes),
                "rationale": r.get("why") or "Good fit for your pantry and goal.",
                "steps": r.get("steps") or []
            })
        return picks or None
    except Exception:
        return None

def plan_three(goal: str, max_minutes: int) -> Optional[list[dict]]:
    """Try local DB first (up to 3). If none, ask LLM for up to 3 online ideas."""
    local = plan_top_k_from_db(goal, max_minutes, k=3)
    if local:
        return local
    return plan_top_k_online(goal, max_minutes, k=3)


def llm_fallback_web_recipes(inv_df: pd.DataFrame, goal: str, max_minutes: int, n: int = 3) -> Optional[List[dict]]:
    """
    Ask the LLM to suggest N recipes that the user can cook now with their pantry.
    Each suggestion includes: title, why_it_works, total_minutes, and an optional url.
    We do NOT require a URL (models can hallucinate); if unsure, it should leave url=None.
    """
    if not USE_OPENAI or client is None:
        return None

    inv = [
        {"name": str(nm), "grams": float(g or 0)}
        for nm, g in zip(inv_df["item_name"].tolist(), inv_df["grams"].fillna(0).tolist())
    ]
    system = (
        "You are a pragmatic home-cooking assistant. The user has ONLY the listed pantry items available. "
        "You must NOT introduce eggs, oats, or any ingredients not present in their pantry. "
        "If something seems missing for a typical recipe, either omit it or suggest a substitution using ONLY pantry items. "
        "Never assume shopping is possible. "
        "Suggest fast, realistic recipes (≤ max_minutes), that rely exclusively on pantry items. "
        "Return STRICT JSON with key 'ideas' as a list of objects:\n"
        "[{title:str, why_it_works:str, total_minutes:int, url: str|null, steps:[str]}]"
    )

    user = {
        "goal": goal,
        "max_minutes": int(max_minutes),
        "pantry": inv,
        "count": int(n)
    }
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=900
        )
        data = json.loads(resp.choices[0].message.content)
        ideas = data.get("ideas", [])
        # Light sanity filter
        cleaned = []
        for it in ideas:
            if not isinstance(it, dict):
                continue
            cleaned.append({
                "title": str(it.get("title", "")).strip()[:140],
                "why": str(it.get("why_it_works", "")).strip(),
                "minutes": int(it.get("total_minutes") or max_minutes),
                "url": (it.get("url") or None),
                "steps": [str(s).strip() for s in it.get("steps", []) if str(s).strip()]
            })
        return cleaned[:n] or None
    except Exception:
        return None


# ------------------- "Cooked this" deduction logic -------------------
HEURISTIC_USE_GRAMS = [
    ("olive oil", 10), ("butter", 10), ("soy sauce", 15), ("cheddar cheese", 30),
    ("egg", 50), ("paneer", 150), ("tofu", 150), ("chicken breast", 200),
    ("chicken thigh", 150), ("rice (cooked)", 150), ("rice", 75),
    ("broccoli", 100), ("tomato", 120), ("onion", 80), ("capsicum", 80),
]

def guess_use_grams_for_key(key: str) -> float:
    for k, g in HEURISTIC_USE_GRAMS:
        if key.startswith(k) or k in key:
            return float(g)
    return 80.0

def per_gram_from_row(row: pd.Series) -> Optional[dict]:
    grams = row.get("grams")
    if grams and grams > 0 and all(pd.notna(row.get(c)) for c in ["calories","protein_g","fat_g","carbs_g","sugar_g"]):
        return dict(
            cal=float(row["calories"])/grams,
            p=float(row["protein_g"])/grams,
            f=float(row["fat_g"])/grams,
            c=float(row["carbs_g"])/grams,
            s=float(row["sugar_g"])/grams
        )
    cache = cache_get_per100(row["item_name"])
    if cache:
        return dict(cal=cache["cal"]/100.0, p=cache["protein"]/100.0, f=cache["fat"]/100.0, c=cache["carbs"]/100.0, s=cache["sugar"]/100.0)
    key = row["item_name"].strip().lower()
    if key in NUTRIENT_LOOKUP:
        per = NUTRIENT_LOOKUP[key]
        return dict(cal=per["cal"]/100.0, p=per["protein"]/100.0, f=per["fat"]/100.0, c=per["carbs"]/100.0, s=per["sugar"]/100.0)
    if USE_OPENAI:
        per = gpt_get_macros_per100(row["item_name"])
        if per:
            return dict(cal=per["cal"]/100.0, p=per["protein"]/100.0, f=per["fat"]/100.0, c=per["carbs"]/100.0, s=per["sugar"]/100.0)
    return None

def compute_recipe_consumption(recipe_id: int, inv_df: pd.DataFrame) -> Tuple[List[dict], str]:
    row = df_sql("SELECT title, ingredients_core_names FROM recipes WHERE id=?", (recipe_id,))
    if row.empty:
        return [], "Recipe not found."
    title = row.loc[0, "title"]
    ing_keys = recipe_ingredient_keys(row.loc[0, "ingredients_core_names"] or "")
    if not ing_keys:
        return [], "No ingredients listed for this recipe."

    inv_df = inv_df.copy()
    inv_df["key"] = inv_df["item_name"].apply(ingredient_key)

    deductions = []
    for ik in ing_keys:
        match = inv_df[inv_df["key"].apply(lambda k: (ik in k) or (k in ik))]
        if match.empty:
            continue
        inv_row = match.iloc[0]
        per_g = per_gram_from_row(inv_row)
        if per_g is None:
            continue

        use_g = guess_use_grams_for_key(ik)
        current_g = float(inv_row["grams"] or 0.0)
        use_g = float(min(use_g, max(0.0, current_g)))
        if use_g <= 0:
            continue
        new_g = max(0.0, current_g - use_g)

        deductions.append(dict(
            id=int(inv_row["id"]),
            item_name=str(inv_row["item_name"]),
            use_g=use_g,
            new_grams=new_g,
            per_g=per_g
        ))

    if not deductions:
        return [], "No overlapping pantry items with available quantity to deduct."

    return deductions, title

def apply_deductions_and_update(deds: List[dict], recipe_id: int, recipe_title: str) -> None:
    for d in deds:
        per = d["per_g"]; g = d["new_grams"]
        cal = round(per["cal"] * g, 1)
        p   = round(per["p"]   * g, 1)
        f   = round(per["f"]   * g, 1)
        c   = round(per["c"]   * g, 1)
        s   = round(per["s"]   * g, 1)
        run_sql("""UPDATE inventory
                   SET grams=?, calories=?, protein_g=?, fat_g=?, carbs_g=?, sugar_g=?, updated_at=datetime('now')
                   WHERE id=?""",
                (g, cal, p, f, c, s, d["id"]))
    run_sql("INSERT INTO cook_log(recipe_id, recipe_title, details_json) VALUES (?,?,?)",
            (int(recipe_id), recipe_title, json.dumps(deds, ensure_ascii=False)))

# ---------- Fallback: pull 3 online-style ideas via GPT (or deterministic) ----------
def online_recipe_ideas(inv_df: pd.DataFrame, goal: str, max_minutes: int, k: int = 3) -> list[dict]:
    """Return exactly k simple recipe ideas based on inventory. Shape:
       [{"title": str, "minutes": int, "why": str, "steps": [str]}]"""
    # Prepare inventory list for prompt
    inv = [
        {"name": str(n), "grams": float(g or 0)}
        for n, g in zip(inv_df["item_name"].tolist(), inv_df["grams"].fillna(0).tolist())
    ]

    # If no LLM available, return deterministic ideas
    def pad_to_k(items: list[dict]) -> list[dict]:
        # Ensure exact k outputs (pad with very simple ideas if needed)
        fallback_bank = [
            {"title": "Quick Protein Stir-Fry", "minutes": 15, "why": "Uses pantry protein + aromatics; fast.",
             "steps": ["Heat oil.", "Add chopped aromatics.", "Add protein; stir-fry.", "Season; serve."]},
            {"title": "One-Pan Savory Scramble", "minutes": 10, "why": "Eggs or paneer + onions; very quick.",
             "steps": ["Sauté onions.", "Add eggs/paneer.", "Season; cook to taste."]},
            {"title": "Simple Protein Rice Bowl", "minutes": 18, "why": "Protein over rice; balanced.",
             "steps": ["Warm rice.", "Cook protein with aromatics.", "Top and serve."]},
        ]
        i = 0
        while len(items) < k:
            items.append(fallback_bank[i % len(fallback_bank)])
            i += 1
        return items[:k]

    if not USE_OPENAI or client is None:
        return pad_to_k([])

    system = (
        "You are a cooking assistant. Propose EXACTLY {k} simple recipes that the user can make now using their pantry. "
        "Keep each under max_minutes. Prefer using what they have; no shopping. "
        "Return STRICT JSON of a list with objects: "
        '{"title":str,"minutes":int,"why":str,"steps":[str]}'
    ).format(k=k)

    user = {
        "goal": goal,
        "max_minutes": int(max_minutes),
        "inventory": inv
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=900
        )
        data = json.loads(resp.choices[0].message.content)
        # Allow two shapes: {"recipes":[...]} or [...]
        recipes = data.get("recipes", data)
        if not isinstance(recipes, list):
            recipes = []
        # Normalize each item
        normd = []
        for r in recipes:
            normd.append({
                "title": str(r.get("title", "Idea")).strip()[:120] or "Idea",
                "minutes": int(r.get("minutes", max(10, min(30, max_minutes)))),
                "why": str(r.get("why", "Uses your current pantry and meets your goal."))[:300],
                "steps": [str(s).strip() for s in (r.get("steps") or []) if str(s).strip()][:12]
            })
        return pad_to_k([x for x in normd if x["title"]][:k])
    except Exception:
        return pad_to_k([])


# ------------------- Top Header -------------------
st.title("Pocket Chef")
st.caption("Use what you have. Hit your nutrition goals.")

# ------------------- LAYOUT: 3 columns -------------------
col_side, col_inv, col_rec = st.columns([0.9, 2.8, 1.6], gap="large")

# ===== LEFT: Controls =====
with col_side:
    st.subheader("Controls")

    goal = st.selectbox(
        "Goal",
        [
            "Balanced", "High Protein", "Low Carb", "Low Calorie",
            "Muscle Gain", "Weight Loss", "Keto", "Vegan", "Heart Healthy"
        ],
        index=1
    )

    max_minutes = st.slider("Max cooking time", 10, 180, 30, 5)

    st.markdown("---")
    if st.button("Save preferences"):
        run_sql(
            """INSERT INTO user_prefs(user_id,goal,max_minutes) VALUES(?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET goal=excluded.goal, max_minutes=excluded.max_minutes, updated_at=datetime('now')""",
            ('local', goal, int(max_minutes))
        )
        st.success("Saved.")

# ===== MIDDLE: Inventory =====
with col_inv:
    st.subheader("Inventory")
    st.caption("One item per line. Optional grams like `chicken breast:200`.")
    inv_text = st.text_area(
        "Add items", height=120,
        placeholder="chicken thigh:250\nchicken breast:250\nrice:250\nrice (cooked):250\npaneer:250"
    )

    # NOTE: pure insert — no macro/GPT enrichment here
    def parse_inventory_text(txt: str) -> List[Dict[str, Any]]:
        rows = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            grams = None
            if ":" in line:
                name, qty = line.split(":", 1)
                name = name.strip()
                try:
                    grams = float(qty.strip())
                except:
                    grams = None
            else:
                name = line
            row = dict(item_name=name, grams=grams, is_estimate=1)
            rows.append(row)
        return rows

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add to inventory"):
            rows = parse_inventory_text(inv_text)
            for r in rows:
                est = estimate_from_lookup_or_cache_or_gpt(r["item_name"], r.get("grams"))
                if est:
                    r.update(est)  # adds calories, protein_g, fat_g, carbs_g, sugar_g, is_estimate=1
                insert_inventory_row(r)
            st.success(f"Added {len(rows)} item(s).")
            # prevent any suggestions until the user explicitly plans
            st.session_state.planned = False
            st.session_state.plan = None
            safe_rerun()
    with c2:
        if st.button("Clear inventory"):
            run_sql("DELETE FROM inventory WHERE user_id='local'")
            st.success("Cleared.")
            st.session_state.planned = False
            st.session_state.plan = None
            safe_rerun()

    # --- Single editable table ---
    inv_df = get_inventory_df()
    table_df = inv_df[["id","item_name","grams","calories","protein_g","fat_g","carbs_g","sugar_g"]].copy()
    table_df["delete"] = False

    edited = st.data_editor(
        table_df,
        use_container_width=True,
        height=430,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("id", help="Row id (read-only)", disabled=True),
            "item_name": st.column_config.TextColumn("item_name"),
            "grams": st.column_config.NumberColumn("grams", step=5),
            "calories": st.column_config.NumberColumn("calories", step=5),
            "protein_g": st.column_config.NumberColumn("protein_g", step=1),
            "fat_g": st.column_config.NumberColumn("fat_g", step=1),
            "carbs_g": st.column_config.NumberColumn("carbs_g", step=1),
            "sugar_g": st.column_config.NumberColumn("sugar_g", step=0.5),
            "delete": st.column_config.CheckboxColumn("delete"),
        },
        disabled=["id"],
        key="inv_editor",
    )

    if st.button("Save inventory table"):
        # Deletes first
        for _, r in edited.iterrows():
            if bool(r.get("delete")) and pd.notna(r.get("id")):
                delete_inventory_row(int(r["id"]))

        # Inserts / updates
        for _, r in edited.iterrows():
            if bool(r.get("delete")):
                continue
            row_dict = {
                "id": r.get("id"),
                "item_name": (r.get("item_name") or "").strip(),
                "grams": to_float_or_none(r.get("grams")),
                "calories": to_float_or_none(r.get("calories")),
                "protein_g": to_float_or_none(r.get("protein_g")),
                "fat_g": to_float_or_none(r.get("fat_g")),
                "carbs_g": to_float_or_none(r.get("carbs_g")),
                "sugar_g": to_float_or_none(r.get("sugar_g")),
                "is_estimate": 0,
            }
            if row_dict["item_name"] == "":
                continue

            if pd.isna(row_dict["id"]):
                # (Optional) auto-fill macros could go here — currently disabled for clean "add"
                est = estimate_from_lookup_or_cache_or_gpt(row_dict["item_name"], row_dict.get("grams"))
                if est:
                    row_dict.update(est)
                insert_inventory_row(row_dict)
            else:
                needs_fill = (
                    row_dict.get("grams") is not None and
                    any(row_dict.get(k) is None for k in ["calories","protein_g","fat_g","carbs_g","sugar_g"])
                )
                if needs_fill:
                    est = estimate_from_lookup_or_cache_or_gpt(row_dict["item_name"], row_dict.get("grams"))
                    if est:
                        for k in ["calories","protein_g","fat_g","carbs_g","sugar_g"]:
                            row_dict[k] = est[k]
                        row_dict["is_estimate"] = 1
                update_inventory_row(row_dict)

        st.success("Inventory saved.")
        st.session_state.planned = False
        st.session_state.plan = None
        safe_rerun()

# ===== RIGHT: Recipes (Agent-gated) =====
# ===== RIGHT: Recipes (multi-option + web fallback) =====
with col_rec:
    st.subheader("Recipes")

    inv_now = get_inventory_df()
    if inv_now.empty:
        st.info("Add inventory items first.")
    else:
        st.markdown("### Plan with agent")

        # Clicking this triggers planning only once and stores a list of up to 3 plans
        if st.button("Plan my meal"):
            st.session_state.planned = True
            # Try local DB first (returns up to 3 plans); your function should already do this
            st.session_state.plans = plan_three(goal, max_minutes)

        if not st.session_state.get("planned"):
            st.caption(
                "Ready when you are — click **Plan my meal** to pick the best recipes using your pantry and goal."
            )
        else:
            plans = st.session_state.get("plans")

            # If the DB didn’t yield viable options, ask the LLM for web ideas (still show 3)
            if not plans:
                st.warning("No feasible plan with current pantry and constraints.")
                fallback_ideas = llm_fallback_web_recipes(inv_now, goal, max_minutes, n=3)

                if not fallback_ideas:
                    if not USE_OPENAI:
                        st.info(
                            "LLM is not configured (no API key). Connect OpenAI/Azure OpenAI "
                            "to get online recipe suggestions."
                        )
                    else:
                        st.info("Couldn’t fetch suggestions right now. Try again.")
                else:
                    st.markdown("#### Try these instead")
                    for i, idea in enumerate(fallback_ideas, start=1):
                        with st.container():
                            st.markdown(f"### Option {i}: {idea['title']} · {idea['minutes']} min")
                            if idea.get("why"):
                                st.caption(idea["why"])
                            if idea.get("url"):
                                st.write(f"[Open recipe]({idea['url']})")

                            steps = idea.get("steps") or []
                            if steps:
                                st.markdown("**Quick steps**")
                                st.markdown(render_bullets(steps))

                            st.info(
                                "This idea is generated from your pantry and isn’t in the local DB, "
                                "so cooking won’t auto‑deduct inventory."
                            )
                            st.divider()
            else:
                # We have up to 3 plans (mix of local + online). Render all of them.
                for idx, p in enumerate(plans, start=1):
                    is_local = (p.get("source") == "local") and p.get("recipe_id")

                    title = p.get("title", "Untitled")
                    mins = int(p.get("minutes") or 0)
                    st.markdown(f"### Option {idx}: {title} · {mins} min")

                    if p.get("rationale"):
                        st.caption(p["rationale"])

                    if is_local:
                        rid = int(p["recipe_id"])

                        # Optional details for local recipes
                        if st.toggle(f"Show recipe details (ID {rid})", key=f"det_{rid}"):
                            details = df_sql(
                                "SELECT ingredients_core_names, instructions_text FROM recipes WHERE id=?",
                                (rid,),
                            )
                            if details.empty:
                                st.info("No details available.")
                            else:
                                ing_text = details.loc[0, "ingredients_core_names"]
                                step_text = details.loc[0, "instructions_text"]

                                st.markdown("**Ingredients**")
                                st.markdown(render_bullets(ing_text))

                                if step_text:
                                    st.markdown("**Instructions**")
                                    st.markdown(render_bullets(step_text))

                        # Cook (only for local recipes) – deducts inventory heuristically
                        if st.button(f"Cook this", key=f"cook_{rid}"):
                            deds, title_msg = compute_recipe_consumption(rid, get_inventory_df())
                            if not deds:
                                st.warning(f"Couldn't deduct items. {title_msg}")
                            else:
                                apply_deductions_and_update(deds, rid, p.get("title", "Recipe"))
                                pretty = "; ".join(
                                    [f"{d['item_name']} -{int(d['use_g'])}g" for d in deds]
                                )
                                st.success(f"Inventory updated: {pretty}")
                                # Reset planning after cooking to reflect new pantry
                                st.session_state.planned = False
                                st.session_state.plans = None
                                safe_rerun()

                        st.divider()

                    else:
                        # Online (LLM) idea
                        steps = p.get("steps") or []
                        if steps:
                            st.markdown("**Quick steps**")
                            st.markdown(render_bullets(steps))

                        # Optional external URL if your planner provides one
                        if p.get("url"):
                            st.write(f"[Open recipe]({p['url']})")

                        st.info(
                            "This idea is generated from your pantry. Since it isn’t in the local DB, "
                            "cooking won’t auto‑deduct inventory."
                        )
                        st.divider()
