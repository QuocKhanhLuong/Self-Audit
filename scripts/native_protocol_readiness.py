"""Report native gates by profile class without data, producer imports or GT access.

Profile classes are reported separately: PAPER_REPRODUCTION (exact original protocol),
PAPER_FAITHFUL_REIMPLEMENTATION (paper-only template; values required),
PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS (paper values plus labelled conventions/fallbacks;
never exact reproduction) and OFFICIAL_REFERENCE (official-code arithmetic). A producer is
RUNNABLE when its protocol lock passes; data availability is reported independently.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_protocol import load_lock, ProtocolBlocked, native_status

CLASSES = ("PAPER_REPRODUCTION", "PAPER_FAITHFUL_REIMPLEMENTATION",
           "PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS", "OFFICIAL_REFERENCE")
DATA = {"DSS_US": {"CAMUS": "BLOCKED_DATA", "DINO_ViT-S/8_checkpoint": "BLOCKED_DATA"},
        "SGSCN": {"PH2": "BLOCKED_DATA", "SYSU-US": "BLOCKED_DATA"}}


def profile_class(config):
    if config.get("profile_class"):
        return config["profile_class"]
    return "OFFICIAL_REFERENCE" if config["profile"].endswith("_official_reference") else "PAPER_REPRODUCTION"


if __name__ == "__main__":
    result = {}
    for name in ("DSS_US", "SGSCN"):
        profiles = {}
        for path in sorted((ROOT / "baseline" / name / "config/native").glob("*.yaml")):
            entry = {"profile_class": profile_class(json.loads(path.read_text(encoding="utf-8")))}
            for purpose in ("producer", "native_track_b"):
                try:
                    load_lock(path, purpose=purpose)
                    entry[purpose] = {"status": "RUNNABLE" if purpose == "producer" else "EVALUATOR_READY"}
                except ProtocolBlocked as error:
                    entry[purpose] = {"status": error.status, "reasons": error.reasons}
            profiles[path.stem] = entry
        by_class = {}
        for cls in CLASSES:
            members = {p: e for p, e in profiles.items() if e["profile_class"] == cls}
            if not members:
                continue
            runnable = sorted(p for p, e in members.items() if e["producer"]["status"] == "RUNNABLE")
            evaluators = sorted(p for p, e in members.items() if e["native_track_b"]["status"] == "EVALUATOR_READY")
            by_class[cls] = {"profiles": len(members),
                             "producer_status": "RUNNABLE" if runnable else "BLOCKED_PROTOCOL",
                             "runnable_profiles": runnable, "track_b_evaluator_ready": evaluators}
        result[name] = {"profiles": profiles, "by_profile_class": by_class, "data": DATA[name],
                        "statuses": native_status(producer_complete=False, track_b_status="BLOCKED_PROTOCOL",
                                                  track_a_status="BLOCKED_ADAPTER", all_required_data=False,
                                                  protocol_resolved=False)}
    print(json.dumps(result, indent=2))
