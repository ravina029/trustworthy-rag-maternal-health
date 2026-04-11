import re

def normalize(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()


def extract_claims(answer):
    # naive but works
    sentences = re.split(r'[.?!]\s+', answer)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def is_claim_supported(claim, chunks):
    claim_norm = normalize(claim)

    for chunk in chunks:
        chunk_norm = normalize(chunk)

        # simple heuristic: keyword overlap
        overlap = set(claim_norm.split()) & set(chunk_norm.split())

        if len(overlap) >= 3:
            return True

    return False


def verify_answer(answer, chunks):
    claims = extract_claims(answer)

    unsupported = []
    for c in claims:
        if not is_claim_supported(c, chunks):
            unsupported.append(c)

    return {
        "all_supported": len(unsupported) == 0,
        "unsupported_claims": unsupported
    }