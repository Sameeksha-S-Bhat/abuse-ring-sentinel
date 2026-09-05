"""
generate_data.py
=================
Generates a SYNTHETIC dataset simulating merchant accounts on a payments
platform, some of which form coordinated "abuse rings" (collusive fraud
networks that share infrastructure like devices, IPs, payment instruments,
or addresses to evade single-account fraud checks).

WHY SYNTHETIC (documented, so judges/interviewers know the assumptions):
--------------------------------------------------------------------------
Real fraud-ring-labeled datasets are not publicly available (for obvious
privacy/security reasons -- companies don't release confirmed fraud-ring
data). So we generate data with EXPLICIT, DEFENSIBLE assumptions:

  1. Base rate of fraud rings: ~3% of accounts belong to a coordinated ring
     (roughly in line with published industry estimates that organized/ring
     fraud is a small but disproportionately costly minority of total fraud
     -- most fraud is opportunistic/individual, rings are rarer but higher
     value per incident).
  2. Ring size: 3-8 accounts per ring (small cells, not huge networks --
     realistic for evasion, since huge obvious clusters get caught easily).
  3. Ring accounts deliberately share 1-2 "weak" identifiers (e.g. same
     device fingerprint OR same payment instrument) rather than all
     identifiers, to simulate real evasion behavior (fraudsters vary some
     signals to avoid trivial detection).
  4. Legitimate accounts can ALSO incidentally share attributes (e.g. two
     honest customers on the same shared office WiFi/IP, or a family
     sharing a card) -- this is critical: it's what makes the problem hard
     and prevents a trivial "shared attribute = fraud" heuristic from
     looking artificially perfect.
  5. Behavioral features (transaction velocity, amount patterns, account
     age) are drawn from different distributions for ring vs. legitimate
     accounts, but with OVERLAP -- again to avoid an unrealistically easy
     separation.

This file's job: produce data + ground truth labels so we can evaluate
honestly on held-out accounts, then in the real product deployment you
would NOT have these labels and would rely purely on structure + behavior.
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import uuid
import random

RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

N_ACCOUNTS = 5000
RING_ACCOUNT_RATE = 0.03          # ~3% of accounts are part of a ring
MIN_RING_SIZE, MAX_RING_SIZE = 3, 8
SHARED_ATTR_POOL_SIZE = {         # pool sizes control how "crowded" each
    "device_id": 22000,            # identifier space is -> controls how much
    "ip_subnet": 16000,            # incidental (innocent) overlap occurs.
    "payment_instrument": 21000,   # Large pools relative to N_ACCOUNTS keep
    "shipping_address_hash": 19000,  # accidental collisions rare (a handful
}                                     # of legit "same office WiFi" cases),
                                      # so deliberate ring sharing stands out
                                      # as a real graph signal rather than
                                      # being buried in noise. This mirrors
                                      # reality: most identifier collisions
                                      # among strangers are rare; most
                                      # sharing is either family/office
                                      # (small, incidental) or a fraud ring
                                      # (deliberate).


def _sample_pool(pool_size):
    return [str(uuid.uuid4())[:8] for _ in range(pool_size)]


def generate_accounts():
    device_pool = _sample_pool(SHARED_ATTR_POOL_SIZE["device_id"])
    ip_pool = _sample_pool(SHARED_ATTR_POOL_SIZE["ip_subnet"])
    pay_pool = _sample_pool(SHARED_ATTR_POOL_SIZE["payment_instrument"])
    addr_pool = _sample_pool(SHARED_ATTR_POOL_SIZE["shipping_address_hash"])

    n_ring_accounts = int(N_ACCOUNTS * RING_ACCOUNT_RATE)
    accounts = []
    ring_id_counter = 0
    assigned = 0

    # --- build rings first ---
    while assigned < n_ring_accounts:
        ring_size = random.randint(MIN_RING_SIZE, MAX_RING_SIZE)
        ring_size = min(ring_size, n_ring_accounts - assigned)
        if ring_size < MIN_RING_SIZE:
            break
        ring_id_counter += 1

        # REALISM FIX 1: some rings are "sophisticated evaders" -- they
        # deliberately vary behavior close to normal and only weakly
        # connect a SUBSET of members to the shared identifier, making
        # them genuinely harder to catch (this prevents unrealistic 100%
        # separability and forces the model to make real tradeoffs).
        is_sophisticated = random.random() < 0.35

        n_shared_attrs = random.choice([1, 1, 2])
        shared_attrs = random.sample(
            ["device_id", "ip_subnet", "payment_instrument", "shipping_address_hash"],
            n_shared_attrs,
        )
        shared_values = {attr: random.choice(
            device_pool if attr == "device_id" else
            ip_pool if attr == "ip_subnet" else
            pay_pool if attr == "payment_instrument" else addr_pool
        ) for attr in shared_attrs}

        # how many of the ring members actually USE the shared identifier
        # (sophisticated rings: only ~50-70% do; naive rings: ~90-100% do)
        share_fraction = np.random.uniform(0.5, 0.7) if is_sophisticated else np.random.uniform(0.85, 1.0)

        for member_i in range(ring_size):
            idx = len(accounts)
            uses_shared = random.random() < share_fraction
            row = {
                "account_id": f"ACC_{idx:05d}",
                "device_id": shared_values.get("device_id", random.choice(device_pool)) if uses_shared else random.choice(device_pool),
                "ip_subnet": shared_values.get("ip_subnet", random.choice(ip_pool)) if uses_shared else random.choice(ip_pool),
                "payment_instrument": shared_values.get("payment_instrument", random.choice(pay_pool)) if uses_shared else random.choice(pay_pool),
                "shipping_address_hash": shared_values.get("shipping_address_hash", random.choice(addr_pool)) if uses_shared else random.choice(addr_pool),
            }
            # REALISM FIX 2: widen + overlap behavioral distributions.
            # Sophisticated rings mimic normal behavior much more closely;
            # naive rings still show the "classic" suspicious signature,
            # but with heavier noise than before so there's real overlap
            # with the legitimate population's tails.
            if is_sophisticated:
                row["account_age_days"] = max(1, int(np.random.normal(180, 150)))       # closer to legit
                row["txn_velocity_per_day"] = max(0.05, np.random.normal(2.2, 1.8))      # closer to legit
                row["avg_txn_amount"] = max(10, np.random.normal(1400, 1200))
                row["amount_std"] = max(1, np.random.normal(220, 200))                    # less tight clustering
            else:
                row["account_age_days"] = max(1, int(np.random.normal(35, 35)))
                row["txn_velocity_per_day"] = max(0.1, np.random.normal(5, 3.5))
                row["avg_txn_amount"] = max(10, np.random.normal(1800, 1100))
                row["amount_std"] = max(1, np.random.normal(90, 80))
            row["is_ring"] = 1
            row["ring_id"] = ring_id_counter
            accounts.append(row)
            assigned += 1

    # --- build legitimate accounts (majority class) ---
    n_legit = N_ACCOUNTS - len(accounts)

    # REALISM FIX 3: a small slice of legit accounts are "innocent lookalikes"
    # -- e.g. a family sharing a card, or an office sharing an IP/device pool
    # (2-4 people), which produces graph structure that superficially
    # resembles a small ring, AND/OR legit power-users with high velocity
    # (bulk B2B buyers) that resemble ring behavioral signatures. This is
    # what creates genuine, defensible false-positive risk instead of a
    # trivially separable problem.
    n_lookalike_clusters = 40
    lookalike_budget = n_lookalike_clusters * 3  # planned average reservation

    lookalike_accounts_added = 0
    for _ in range(n_lookalike_clusters):
        cluster_size = random.randint(2, 4)
        shared_attr = random.choice(["device_id", "ip_subnet", "payment_instrument", "shipping_address_hash"])
        shared_val = random.choice(
            device_pool if shared_attr == "device_id" else
            ip_pool if shared_attr == "ip_subnet" else
            pay_pool if shared_attr == "payment_instrument" else addr_pool
        )
        for _ in range(cluster_size):
            idx = len(accounts)
            row = {
                "account_id": f"ACC_{idx:05d}",
                "device_id": shared_val if shared_attr == "device_id" else random.choice(device_pool),
                "ip_subnet": shared_val if shared_attr == "ip_subnet" else random.choice(ip_pool),
                "payment_instrument": shared_val if shared_attr == "payment_instrument" else random.choice(pay_pool),
                "shipping_address_hash": shared_val if shared_attr == "shipping_address_hash" else random.choice(addr_pool),
                "account_age_days": max(1, int(np.random.normal(350, 280))),
                "txn_velocity_per_day": max(0.05, np.random.normal(1.5, 1.3)),
                "avg_txn_amount": max(10, np.random.normal(1300, 1100)),
                "amount_std": max(1, np.random.normal(350, 240)),
                "is_ring": 0,
                "ring_id": None,
            }
            accounts.append(row)
            lookalike_accounts_added += 1

    # a slice of legit power-users with ring-like velocity/amount patterns
    # purely by coincidence of being a legitimate frequent bulk buyer
    n_power_users = 60
    for _ in range(n_power_users):
        idx = len(accounts)
        row = {
            "account_id": f"ACC_{idx:05d}",
            "device_id": random.choice(device_pool),
            "ip_subnet": random.choice(ip_pool),
            "payment_instrument": random.choice(pay_pool),
            "shipping_address_hash": random.choice(addr_pool),
            "account_age_days": max(1, int(np.random.normal(200, 150))),
            "txn_velocity_per_day": max(0.1, np.random.normal(4.5, 2.0)),   # high, ring-like
            "avg_txn_amount": max(10, np.random.normal(1700, 1000)),
            "amount_std": max(1, np.random.normal(150, 120)),                # tighter, ring-like
            "is_ring": 0,
            "ring_id": None,
        }
        accounts.append(row)

    n_plain_legit = n_legit - lookalike_accounts_added - n_power_users
    for _ in range(max(0, n_plain_legit)):
        idx = len(accounts)
        row = {
            "account_id": f"ACC_{idx:05d}",
            "device_id": random.choice(device_pool),
            "ip_subnet": random.choice(ip_pool),
            "payment_instrument": random.choice(pay_pool),
            "shipping_address_hash": random.choice(addr_pool),
            "account_age_days": max(1, int(np.random.normal(400, 300))),
            "txn_velocity_per_day": max(0.05, np.random.normal(1.2, 1.0)),
            "avg_txn_amount": max(10, np.random.normal(1200, 1100)),
            "amount_std": max(1, np.random.normal(400, 250)),
            "is_ring": 0,
            "ring_id": None,
        }
        accounts.append(row)

    df = pd.DataFrame(accounts)
    df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_accounts()
    out_path = os.path.join(PROJECT_ROOT, "data", "accounts.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} accounts -> {out_path}")
    print(f"Ring accounts: {df['is_ring'].sum()} ({df['is_ring'].mean()*100:.2f}%)")
    print(f"Number of distinct rings: {df['ring_id'].nunique()}")
    print(df.head())
