"""
Pakistan Constitution Q&A Data Pipeline - FULL VERSION
Extracts ALL articles from the Constitution, not just Fundamental Rights
"""

import re
import json
import random
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR    = r"D:\pakistan_constitution_llm"
RAW_TXT     = f"{BASE_DIR}\\data\\raw\\constitution.txt"
OUT_CLEANED = f"{BASE_DIR}\\data\\cleaned\\constitution.txt"
OUT_CHUNKS  = f"{BASE_DIR}\\data\\chunks\\articles.json"
OUT_QA      = f"{BASE_DIR}\\data\\qa_pairs\\constitution_qa.jsonl"
OUT_TRAIN   = f"{BASE_DIR}\\data\\splits\\constitution_train.jsonl"
OUT_VAL     = f"{BASE_DIR}\\data\\splits\\constitution_validation.jsonl"
OUT_TEST    = f"{BASE_DIR}\\data\\splits\\constitution_test.jsonl"

for p in [OUT_CLEANED, OUT_CHUNKS, OUT_QA, OUT_TRAIN, OUT_VAL, OUT_TEST]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# ── ALL Article titles in the Constitution ─────────────────────────
# Part II: Fundamental Rights and Principles of Policy (Articles 8-40)
# Part III: The Federation of Pakistan (Articles 41-100)
# And beyond...
ALL_ARTICLES = {
    # Part II - Fundamental Rights
    "8": "Laws inconsistent with or in derogation of Fundamental Rights to be void",
    "9": "Security of person",
    "10": "Safeguards as to arrest and detention",
    "10A": "Right to fair trial",
    "11": "Slavery, forced labour, etc., prohibited",
    "12": "Protection against retrospective punishment",
    "13": "Protection against double punishment and self incrimination",
    "14": "Inviolability of dignity of man",
    "15": "Freedom of movement",
    "16": "Freedom of assembly",
    "17": "Freedom of association",
    "18": "Freedom of trade, business or profession",
    "19": "Freedom of speech",
    "19A": "Right to information",
    "20": "Freedom to profess religion and to manage religious institutions",
    "21": "Safeguard against taxation for purposes of any particular religion",
    "22": "Safeguards as to educational institutions in respect of religion",
    "23": "Provision as to property",
    "24": "Protection of property rights",
    "25": "Equality of citizens",
    "25A": "Right to education",
    "26": "Non-discrimination in respect of access to public places",
    "27": "Safeguard against discrimination in services",
    "28": "Preservation of language, script and culture",
    "29": "Principles of Policy",
    "30": "Responsibility with respect to Principles of Policy",
    "31": "Islamic way of life",
    "32": "Promotion of local government institutions",
    "33": "Parochial and other similar prejudices to be discouraged",
    "34": "Full participation of women in national life",
    "35": "Protection of family, etc.",
    "36": "Protection of minorities",
    "37": "Promotion of social justice and eradication of social evils",
    "38": "Promotion of social and economic well-being of the people",
    "39": "Participation of people in Armed Forces",
    "40": "Strengthening bonds with Muslim world and promoting international peace",
    
    # Part III - The Federation (selected key articles)
    "41": "The President",
    "43": "Conditions of President's office",
    "44": "Term of office of President",
    "45": "President's power to grant pardon, etc.",
    "46": "President to be kept informed",
    "47": "Removal or impeachment of President",
    "48": "President to act on advice, etc.",
    "50": "Majlis-e-Shoora (Parliament)",
    "51": "National Assembly",
    "52": "Duration of National Assembly",
    "53": "Speaker and Deputy Speaker of National Assembly",
    "54": "Summoning and prorogation of Majlis-e-Shoora",
    "55": "Voting in Assembly and quorum",
    "56": "Address by President",
    "58": "Dissolution of National Assembly",
    "59": "The Senate",
    "60": "Chairman and Deputy Chairman of Senate",
    "62": "Qualifications for membership of Majlis-e-Shoora",
    "63": "Disqualifications for membership of Majlis-e-Shoora",
    "63A": "Disqualification on grounds of defection",
    "91": "The Cabinet",
    "95": "Voting on account",
    
    # Part IV - Provinces
    "101": "Governor of a Province",
    "105": "Speaker of Provincial Assembly",
    "128": "Provincial Government",
    "130": "The Cabinet",
    
    # Part V - Relations between Federation and Provinces
    "141": "Extent of executive authority of Federation",
    "142": "Subject-matter of Federal and Provincial laws",
    "143": "Inconsistency between Federal and Provincial laws",
    
    # Part VI - Finance, Property, Contracts and Suits
    "160": "National Finance Commission",
    "161": "Natural gas and hydro-electric power",
    
    # Part VII - The Judicature
    "175": "Establishment and jurisdiction of courts",
    "176": "Constitution of Supreme Court",
    "177": "Appointment of Supreme Court judges",
    "184": "Original jurisdiction of Supreme Court",
    "185": "Appellate jurisdiction of Supreme Court",
    "187": "Issue and execution of processes of Supreme Court",
    "188": "Review of judgments or orders by the Supreme Court",
    "199": "Jurisdiction of High Court",
    
    # Part VIII - Elections
    "213": "Chief Election Commissioner",
    "218": "Election Commission",
    "219": "Duties of Election Commission",
    
    # Part IX - Islamic Provisions
    "227": "Provisions relating to the Holy Quran and Sunnah",
    "228": "Composition, etc., of Islamic Council",
    "230": "Functions of Islamic Council",
    
    # Part X - Emergency Provisions
    "232": "Proclamation of emergency on account of war, etc.",
    "233": "Power to suspend Fundamental Rights during emergency",
    "234": "Power to issue Proclamation in case of failure of Constitutional machinery in a Province",
    "236": "Revocation of Proclamation, etc.",
    
    # Part XI - Amendment of Constitution
    "238": "Amendment of Constitution",
    
    # Part XII - Miscellaneous
    "240": "Indemnity",
    "242": "Terms and conditions of service of judges",
    "243": "Armed Forces",
    "245": "Functions of Armed Forces",
    "248": "Protection to President, Governor, etc.",
    "251": "National language",
    "255": "Oath of office",
    "257": "Functions of Majlis-e-Shoora in relation to Azad Jammu and Kashmir",
    "260": "Definitions",
}

# ── STEP 1: Clean text ─────────────────────────────────────────────
def clean_text(raw: str) -> str:
    text = raw.replace("\f", "\n")
    text = re.sub(r'\n\d+\s+(Subs\.|Ins\.|Added|Omitted|See|New Article)[^\n]+', '', text)
    text = re.sub(r'\nCONSTITUTION OF PAKISTAN\s*\n', '\n', text)
    text = re.sub(r'[^\n]+\.{5,}\s*\d+\s*\n', '', text)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\d+\[', '[', text)
    return text.strip()

# ── STEP 2: Extract ALL articles ───────────────────────────────────
def extract_all_articles(text: str) -> dict:
    """
    Extract ALL articles found in the text, not just predefined ones.
    Uses regex to find any article number pattern.
    """
    articles = {}
    
    # Pattern: article number at start of line, followed by title
    # Matches: "8.", "10A.", "41.", "63A.", "160.", etc.
    pattern = r'(?:^|\n)\s*(\d+[A-Z]?)\.\s*([^\n]+?)(?:\n|$)'
    
    matches = list(re.finditer(pattern, text))
    
    for i, m in enumerate(matches):
        num = m.group(1)
        title = m.group(2).strip()
        
        # Skip if title is just a number or too short
        if len(title) < 5 or title.isdigit():
            continue
            
        # Clean title
        title = re.sub(r'\s+', ' ', title)
        title = title.replace('[', '').replace(']', '')
        
        # Find body: from end of title to start of next article
        start_pos = m.end()
        
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
            body = text[start_pos:end_pos]
        else:
            body = text[start_pos:]
        
        # Clean body
        body = re.sub(r'\s+', ' ', body).strip()
        body = body.replace('[', '').replace(']', '')
        body = re.sub(r'\d+(Subs|Ins|Added|See|New)[^.]+\.', '', body)
        
        # Use known title if available, otherwise use extracted
        known_title = ALL_ARTICLES.get(num)
        if known_title:
            title = known_title
        
        if len(body) > 50:
            articles[num] = {
                "title": title,
                "body": body[:2500]  # Slightly larger for complex articles
            }
    
    return articles

# ── STEP 3: Generate Q&A pairs for ALL topics ──────────────────────
def clean_body(body: str) -> str:
    b = re.sub(r'\s+', ' ', body).strip()
    b = re.sub(r'(Subs|Ins|Added|Omitted|See footnote)[^.]+\.', '', b)
    b = b.strip()
    if len(b) > 800:
        cut = b[:800].rfind('.')
        b = b[:cut+1] if cut > 100 else b[:800]
    return b

def generate_qa_all(articles: dict) -> list:
    pairs = []
    
    for num, data in articles.items():
        title = data["title"]
        body = clean_body(data["body"])
        art = f"Article {num}"
        
        if not body or len(body) < 30:
            continue
        
        # Determine article category for better questions
        category = categorize_article(num)
        
        # Template 1: Direct article question
        pairs.append({
            "question": f"What does {art} of the Pakistan Constitution state?",
            "answer": f"{art} of the Pakistan Constitution deals with {title}. {body}"
        })
        
        # Template 2: What is it about
        pairs.append({
            "question": f"What is {art} of the Constitution of Pakistan about?",
            "answer": f"{art} is about {title}. {body}"
        })
        
        # Template 3: Category-specific questions
        cat_questions = get_category_questions(num, title, body, art, category)
        pairs.extend(cat_questions)
        
        # Template 4: General knowledge
        pairs.append({
            "question": f"Under the Pakistan Constitution, what does {title.lower()} mean?",
            "answer": f"The Pakistan Constitution addresses {title.lower()} under {art}: {body}"
        })
    
    return pairs

def categorize_article(num: str) -> str:
    """Categorize article by number."""
    try:
        n = int(re.match(r'\d+', num).group())
    except:
        return "general"
    
    if 8 <= n <= 28:
        return "fundamental_rights"
    elif 29 <= n <= 40:
        return "principles_of_policy"
    elif 41 <= n <= 100:
        return "federation"
    elif 101 <= n <= 127:
        return "provinces"
    elif 128 <= n <= 140:
        return "provincial_government"
    elif 141 <= n <= 159:
        return "federal_provincial_relations"
    elif 160 <= n <= 174:
        return "finance"
    elif 175 <= n <= 212:
        return "judiciary"
    elif 213 <= n <= 226:
        return "elections"
    elif 227 <= n <= 231:
        return "islamic_provisions"
    elif 232 <= n <= 237:
        return "emergency"
    elif n >= 238:
        return "miscellaneous"
    return "general"

def get_category_questions(num: str, title: str, body: str, art: str, category: str) -> list:
    """Generate category-specific questions."""
    questions = []
    
    if category == "fundamental_rights":
        if any(w in title.lower() for w in ["right", "freedom", "security", "protection", "safeguard"]):
            questions.append({
                "question": f"What rights does {art} of the Pakistan Constitution guarantee?",
                "answer": f"Under {art}, which concerns {title}: {body}"
            })
            questions.append({
                "question": f"As a Pakistani citizen, what protection does {art} provide me?",
                "answer": f"{art} ({title}) provides the following protection: {body}"
            })
    
    elif category == "federation":
        if "president" in title.lower():
            questions.append({
                "question": f"What does the Pakistan Constitution say about the President regarding {title.lower()}?",
                "answer": f"{art} states: {body}"
            })
        elif "parliament" in title.lower() or "assembly" in title.lower():
            questions.append({
                "question": f"What does {art} say about Parliament?",
                "answer": f"{art} ({title}): {body}"
            })
        elif "speaker" in title.lower():
            questions.append({
                "question": f"What are the provisions for the Speaker under {art}?",
                "answer": f"{art} states: {body}"
            })
    
    elif category == "judiciary":
        questions.append({
            "question": f"What does {art} say about the judiciary regarding {title.lower()}?",
            "answer": f"{art} states: {body}"
        })
        if "supreme court" in title.lower():
            questions.append({
                "question": f"What are the powers of the Supreme Court under {art}?",
                "answer": f"{art} ({title}): {body}"
            })
    
    elif category == "emergency":
        questions.append({
            "question": f"What does {art} say about emergency provisions?",
            "answer": f"{art} ({title}): {body}"
        })
    
    elif category == "islamic_provisions":
        questions.append({
            "question": f"What does {art} say about Islamic provisions in the Constitution?",
            "answer": f"{art} ({title}): {body}"
        })
    
    elif category == "elections":
        questions.append({
            "question": f"What does {art} say about elections in Pakistan?",
            "answer": f"{art} ({title}): {body}"
        })
    
    elif category == "finance":
        questions.append({
            "question": f"What does {art} say about financial matters in Pakistan?",
            "answer": f"{art} ({title}): {body}"
        })
    
    elif category == "provinces":
        questions.append({
            "question": f"What does {art} say about provincial matters in Pakistan?",
            "answer": f"{art} ({title}): {body}"
        })
    
    # Add restriction questions for any article with "subject to" or "restriction"
    if "subject to" in body.lower() or "restriction" in body.lower():
        questions.append({
            "question": f"What restrictions or conditions apply to {title.lower()} under {art}?",
            "answer": f"{art} allows {title.lower()} under certain conditions: {body}"
        })
    
    return questions

# ── STEP 4: Split & Save ───────────────────────────────────────────
def split_dataset(pairs: list):
    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return pairs[:n_train], pairs[n_train:n_train+n_val], pairs[n_train+n_val:]

def save_jsonl(data: list, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ── MAIN ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Pakistan Constitution Q&A Pipeline - FULL EDITION")
    print("Extracts ALL articles from the Constitution")
    print("=" * 60)

    print("\n[1/4] Cleaning text...")
    with open(RAW_TXT, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    cleaned = clean_text(raw)
    with open(OUT_CLEANED, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f"      Cleaned: {len(raw):,} → {len(cleaned):,} chars")

    print("\n[2/4] Extracting ALL articles...")
    articles = extract_all_articles(cleaned)
    
    # Report what we found
    categories = {}
    for num, data in articles.items():
        cat = categorize_article(num)
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n      Total extracted: {len(articles)} articles")
    print("\n      By category:")
    for cat, count in sorted(categories.items()):
        print(f"        {cat}: {count}")
    
    # Check which known articles we missed
    found = set(articles.keys())
    expected = set(ALL_ARTICLES.keys())
    missing = expected - found
    if missing:
        print(f"\n      Missing known articles: {sorted(missing)}")

    with open(OUT_CHUNKS, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print("\n[3/4] Generating Q&A pairs...")
    pairs = generate_qa_all(articles)
    
    # Deduplicate
    seen = set()
    unique = []
    for p in pairs:
        key = p["question"].lower().strip()
        if key not in seen and len(p["answer"]) > 30:
            seen.add(key)
            unique.append(p)
    pairs = unique
    
    save_jsonl(pairs, OUT_QA)
    print(f"      Generated: {len(pairs)} unique Q&A pairs")

    print("\n[4/4] Splitting dataset...")
    train, val, test = split_dataset(pairs)
    save_jsonl(train, OUT_TRAIN)
    save_jsonl(val, OUT_VAL)
    save_jsonl(test, OUT_TEST)
    print(f"      Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    print("\n── Sample Q&A (different categories) ─────────────────")
    samples = []
    for p in pairs:
        m = re.search(r'Article (\d+[A-Z]?)', p['answer'])
        if m:
            cat = categorize_article(m.group(1))
            if cat not in [s.get('cat') for s in samples]:
                samples.append({'cat': cat, 'q': p})
        if len(samples) >= 5:
            break
    
    for s in samples:
        print(f"\n[{s['cat']}]")
        print(f"Q: {s['q']['question']}")
        print(f"A: {s['q']['answer'][:150]}...")

    print("\n✓ Full pipeline complete!")

if __name__ == "__main__":
 main()