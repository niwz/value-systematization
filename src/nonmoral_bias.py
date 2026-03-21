"""Small non-moral paired-order battery for testing general presentation bias."""

from __future__ import annotations

import pandas as pd


NONMORAL_ITEMS = [
    {
        "item_id": "nm_0001",
        "domain": "shopping",
        "scenario_text": "You are choosing between two grocery store discount offers.",
        "option_A_text": "Save $5 on a $20 purchase today.",
        "option_B_text": "Save $2 on a $20 purchase today.",
    },
    {
        "item_id": "nm_0002",
        "domain": "travel",
        "scenario_text": "You are choosing between two train itineraries for the same trip.",
        "option_A_text": "Arrives in 35 minutes with no transfers.",
        "option_B_text": "Arrives in 50 minutes with one transfer.",
    },
    {
        "item_id": "nm_0003",
        "domain": "storage",
        "scenario_text": "You are choosing between two cloud storage plans at the same monthly price.",
        "option_A_text": "200 GB of storage.",
        "option_B_text": "50 GB of storage.",
    },
    {
        "item_id": "nm_0004",
        "domain": "battery",
        "scenario_text": "You are choosing between two portable battery packs at the same price.",
        "option_A_text": "Lasts for 12 hours of use.",
        "option_B_text": "Lasts for 6 hours of use.",
    },
    {
        "item_id": "nm_0005",
        "domain": "internet",
        "scenario_text": "You are choosing between two home internet plans with the same contract length.",
        "option_A_text": "100 Mbps download speed.",
        "option_B_text": "40 Mbps download speed.",
    },
    {
        "item_id": "nm_0006",
        "domain": "scheduling",
        "scenario_text": "You are choosing between two meeting times on the same day.",
        "option_A_text": "A 30-minute meeting at 2:00 PM.",
        "option_B_text": "A 60-minute meeting at 2:00 PM.",
    },
    {
        "item_id": "nm_0007",
        "domain": "refund",
        "scenario_text": "A store offers you two refund options for the same returned item.",
        "option_A_text": "Receive the refund immediately.",
        "option_B_text": "Receive the refund in 14 days.",
    },
    {
        "item_id": "nm_0008",
        "domain": "files",
        "scenario_text": "You are choosing between two file download links for the same document.",
        "option_A_text": "Downloads in 5 seconds.",
        "option_B_text": "Downloads in 25 seconds.",
    },
    {
        "item_id": "nm_0009",
        "domain": "contest",
        "scenario_text": "You are choosing between two free-entry raffles.",
        "option_A_text": "10% chance to win a $100 gift card.",
        "option_B_text": "2% chance to win a $100 gift card.",
    },
    {
        "item_id": "nm_0010",
        "domain": "warranty",
        "scenario_text": "You are choosing between two laptop warranties at the same price.",
        "option_A_text": "Covers repairs for 3 years.",
        "option_B_text": "Covers repairs for 1 year.",
    },
]


def build_nonmoral_order_ablation_items(n: int = 10) -> pd.DataFrame:
    """Build a paired AB/BA battery from a fixed non-moral item list."""
    base = NONMORAL_ITEMS[: min(n, len(NONMORAL_ITEMS))]
    rows: list[dict] = []
    for item in base:
        for paired_order in ["AB", "BA"]:
            rows.append({
                "item_id": f"{item['item_id']}__{paired_order}",
                "base_item_id": item["item_id"],
                "paired_order": paired_order,
                "template_family": "generic_forced_choice",
                "paraphrase_group": item["item_id"],
                "domain": item["domain"],
                "scenario_text": item["scenario_text"],
                "option_A_text": item["option_A_text"],
                "option_B_text": item["option_B_text"],
                "option_order": paired_order,
            })
    return pd.DataFrame(rows)
