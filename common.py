"""Shared data + helpers for the Qdrant assignment."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "company_docs"
MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim
DIM = 384

# (text, source, page, category) -> 30 docs; exactly 4 are HR
RAW = [
    # HR (4)
    ("Employees get 24 days of paid annual leave per year.", "hr_policy.pdf", 1, "HR"),
    ("New joiners complete onboarding within the first two weeks.", "hr_policy.pdf", 3, "HR"),
    ("Performance reviews are held twice a year in June and December.", "hr_policy.pdf", 4, "HR"),
    ("Maternity leave is 26 weeks with full pay.", "hr_policy.pdf", 5, "HR"),
    # IT (8)
    ("Passwords must be at least 12 characters and rotated every 90 days.", "it_security.pdf", 1, "IT"),
    ("Enable two-factor authentication on all company accounts.", "it_security.pdf", 2, "IT"),
    ("Laptops must use full disk encryption.", "it_security.pdf", 3, "IT"),
    ("Report phishing emails to the security team immediately.", "it_security.pdf", 4, "IT"),
    ("VPN is required when accessing internal systems remotely.", "it_security.pdf", 5, "IT"),
    ("Software installs need approval from the IT helpdesk.", "it_handbook.pdf", 2, "IT"),
    ("Backups of production databases run every night at 2 AM.", "it_handbook.pdf", 3, "IT"),
    ("Raise a ticket on the helpdesk portal for hardware issues.", "it_handbook.pdf", 6, "IT"),
    # Finance (8)
    ("Expense claims must be submitted within 30 days with receipts.", "finance.pdf", 1, "Finance"),
    ("Travel above 500 USD needs manager pre-approval.", "finance.pdf", 2, "Finance"),
    ("Salaries are credited on the last working day of the month.", "finance.pdf", 3, "Finance"),
    ("Invoices are paid within 45 days of receipt.", "finance.pdf", 4, "Finance"),
    ("The financial year runs from April to March.", "finance.pdf", 5, "Finance"),
    ("Corporate cards may only be used for business expenses.", "finance.pdf", 6, "Finance"),
    ("Quarterly budgets are reviewed by the finance committee.", "budget.pdf", 2, "Finance"),
    ("Petty cash limit per transaction is 50 USD.", "budget.pdf", 4, "Finance"),
    # Legal (5)
    ("All vendor contracts must be reviewed by the legal team.", "legal.pdf", 1, "Legal"),
    ("Employees sign a non-disclosure agreement on joining.", "legal.pdf", 2, "Legal"),
    ("Customer data is processed according to GDPR requirements.", "legal.pdf", 3, "Legal"),
    ("Trademarks and logos may not be used without written consent.", "legal.pdf", 5, "Legal"),
    ("Data breaches must be reported within 72 hours.", "legal.pdf", 6, "Legal"),
    # Ops (5)
    ("Office hours are 9 AM to 6 PM, Monday to Friday.", "ops.pdf", 1, "Ops"),
    ("Visitors must register at reception and wear a badge.", "ops.pdf", 2, "Ops"),
    ("Fire drills are conducted every six months.", "ops.pdf", 3, "Ops"),
    ("Meeting rooms are booked through the shared calendar.", "ops.pdf", 4, "Ops"),
    ("Cafeteria is open from 12 PM to 3 PM on weekdays.", "ops.pdf", 5, "Ops"),
]
assert len(RAW) == 30


def get_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)


def build_points(model, suffix=""):
    """Stable integer ids (1..30) so re-running/re-upserting overwrites, not duplicates."""
    texts = [r[0] + suffix for r in RAW]
    vectors = model.encode(texts, normalize_embeddings=True)
    return [
        PointStruct(
            id=i + 1,
            vector=vectors[i].tolist(),
            payload={"text": texts[i], "source": RAW[i][1], "page": RAW[i][2], "category": RAW[i][3]},
        )
        for i in range(len(RAW))
    ]


def recreate_collection(client, name, dim, distance=Distance.COSINE):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=dim, distance=distance))


def show(results, title=""):
    if title:
        print(f"\n{title}")
    for p in results:
        print(f"  id={p.id:<3} score={p.score:.4f}  payload={p.payload}")
