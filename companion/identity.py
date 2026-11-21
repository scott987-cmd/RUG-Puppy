"""Local matching against explicitly enrolled household references.

This returns a conversational identity hint, never parent authorization. Engines do
not provide liveness or replay detection. Callers must authenticate enrollment and
revocation, and use only the top-level person_id when selecting personal memory.
"""
import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


class QualityError(ValueError):
    """Input cannot support an identity decision; its code is safe to show in UI."""


@dataclass(frozen=True)
class Observation:
    modality: str
    model_id: str
    vector: list[float]
    quality: dict[str, float] = field(default_factory=dict)
    captured_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class IdentityPolicy:
    # Conservative starting points, not a measured household/child accuracy claim.
    face_threshold: float = 0.60
    voice_threshold: float = 0.75
    candidate_margin: float = 0.10
    min_samples: int = 2
    max_samples: int = 5
    max_age_seconds: float = 15.0
    max_modality_gap_seconds: float = 8.0

    def __post_init__(self):
        if not (0 < self.face_threshold <= 1 and 0 < self.voice_threshold <= 1
                and 0 < self.candidate_margin < 1 and 2 <= self.min_samples <= self.max_samples <= 10
                and 0 < self.max_age_seconds <= 60 and 0 < self.max_modality_gap_seconds <= 30):
            raise ValueError("invalid_identity_policy")


def normalized(vector) -> list[float]:
    values = [float(value) for value in vector]
    if not 2 <= len(values) <= 1024 or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid_embedding")
    length = math.sqrt(sum(value * value for value in values))
    if length < 1e-10:
        raise ValueError("empty_embedding")
    return [value / length for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding_dimension_mismatch")
    return sum(a * b for a, b in zip(left, right))


class IdentityStore:
    """Embeddings and sample metadata only; raw face/audio bytes are not stored."""

    def __init__(self, database: str | Path):
        path = Path(database)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not path.exists():
            path.touch(mode=0o600)
        path.chmod(0o600)
        self.database = str(path)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS identity_samples (
                sample_id TEXT PRIMARY KEY, person_id TEXT NOT NULL,
                modality TEXT NOT NULL, model_id TEXT NOT NULL,
                embedding TEXT NOT NULL, quality TEXT NOT NULL, created_at REAL NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS identity_member ON identity_samples(person_id, modality)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _parent(parent_confirmed: bool):
        if parent_confirmed is not True:
            raise PermissionError("parent_confirmation_required")

    def enroll(self, person_id: str, observation: Observation, *, parent_confirmed: bool = False,
               max_samples: int = 5) -> dict:
        self._parent(parent_confirmed)
        if not isinstance(person_id, str) or not person_id.strip() or len(person_id) > 128:
            raise ValueError("invalid_person_id")
        if observation.modality not in {"face", "voice"} or not observation.model_id:
            raise ValueError("invalid_observation")
        vector = normalized(observation.vector)
        sample_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("""SELECT model_id, embedding FROM identity_samples
                WHERE person_id=? AND modality=?""", (person_id, observation.modality)).fetchall()
            if len(existing) >= max_samples:
                raise ValueError("sample_limit_revoke_one_first")
            for row in existing:
                if row["model_id"] != observation.model_id:
                    raise ValueError("model_changed_reenroll_required")
                reference = json.loads(row["embedding"])
                if cosine(vector, reference) > 0.99999:
                    raise ValueError("duplicate_sample")
            db.execute("INSERT INTO identity_samples VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (sample_id, person_id, observation.modality, observation.model_id,
                        json.dumps(vector), json.dumps(observation.quality), time.time()))
        return {"sample_id": sample_id, "person_id": person_id,
                "modality": observation.modality, "sample_count": len(existing) + 1}

    def list_samples(self, person_id: str | None = None) -> list[dict]:
        with self.connect() as db:
            sql = "SELECT sample_id, person_id, modality, model_id, quality, created_at FROM identity_samples"
            rows = db.execute(sql + (" WHERE person_id=?" if person_id is not None else "")
                              + " ORDER BY created_at", (person_id,) if person_id is not None else ()).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["quality"] = json.loads(value["quality"])
            result.append(value)
        return result

    def references(self, modality: str, model_id: str) -> dict[str, list[list[float]]]:
        with self.connect() as db:
            rows = db.execute("SELECT person_id, embedding FROM identity_samples WHERE modality=? AND model_id=?",
                              (modality, model_id)).fetchall()
        grouped = {}
        for row in rows:
            grouped.setdefault(row["person_id"], []).append(json.loads(row["embedding"]))
        return grouped

    def revoke_sample(self, sample_id: str, *, parent_confirmed: bool = False) -> bool:
        self._parent(parent_confirmed)
        with self.connect() as db:
            return db.execute("DELETE FROM identity_samples WHERE sample_id=?", (sample_id,)).rowcount > 0

    def revoke_person(self, person_id: str, *, parent_confirmed: bool = False) -> int:
        self._parent(parent_confirmed)
        with self.connect() as db:
            return db.execute("DELETE FROM identity_samples WHERE person_id=?", (person_id,)).rowcount


class IdentityService:
    def __init__(self, store: IdentityStore, face_engine=None, voice_engine=None,
                 policy: IdentityPolicy | None = None):
        self.store, self.face_engine, self.voice_engine = store, face_engine, voice_engine
        self.policy = policy or IdentityPolicy()

    def enroll(self, person_id: str, modality: str, data: bytes, *, parent_confirmed: bool = False) -> dict:
        # Reject before running a model or retaining any biometric feature.
        self.store._parent(parent_confirmed)
        observation = self.prepare_enrollment(modality, data)
        return self.commit_enrollment(person_id, observation, parent_confirmed=True)

    def prepare_enrollment(self, modality: str, data: bytes) -> Observation:
        """Extract without persisting, so callers can recheck expiring authorization."""
        return self._extract(modality, data, time.time())

    def commit_enrollment(self, person_id: str, observation: Observation, *,
                          parent_confirmed: bool = False) -> dict:
        self.store._parent(parent_confirmed)
        modality = observation.modality
        existing = self.store.references(modality, observation.model_id).get(person_id, [])
        if existing:
            center = normalized([sum(parts) for parts in zip(*existing)])
            threshold = self.policy.face_threshold if modality == "face" else self.policy.voice_threshold
            if cosine(normalized(observation.vector), center) < threshold:
                raise QualityError("enrollment_inconsistent_retake_sample")
        return self.store.enroll(person_id, observation, parent_confirmed=True,
                                 max_samples=self.policy.max_samples)

    def _extract(self, modality: str, data: bytes, captured_at: float) -> Observation:
        engine = {"face": self.face_engine, "voice": self.voice_engine}.get(modality)
        if engine is None:
            raise QualityError("engine_unavailable")
        return engine.extract(data, captured_at=captured_at)

    def match(self, observation: Observation) -> dict:
        if observation.modality not in {"face", "voice"}:
            return {"person_id": None, "reason": "invalid_modality"}
        age = time.time() - observation.captured_at
        if not math.isfinite(age) or age < -2 or age > self.policy.max_age_seconds:
            return {"person_id": None, "reason": "stale_observation"}
        query = normalized(observation.vector)
        references = self.store.references(observation.modality, observation.model_id)
        scored = []
        for person_id, samples in references.items():
            if len(samples) < self.policy.min_samples:
                continue
            center = normalized([sum(parts) for parts in zip(*samples)])
            scored.append((cosine(query, center), person_id))
        if not scored:
            return {"person_id": None, "reason": "insufficient_enrollment"}
        scored.sort(reverse=True)
        threshold = (self.policy.face_threshold if observation.modality == "face"
                     else self.policy.voice_threshold)
        score, person_id = scored[0]
        if score < threshold:
            return {"person_id": None, "reason": "below_threshold", "score": score}
        if len(scored) > 1 and (scored[1][0] >= threshold
                               or score - scored[1][0] < self.policy.candidate_margin):
            return {"person_id": None, "reason": "ambiguous_candidates", "score": score}
        return {"person_id": person_id, "reason": "matched", "score": score,
                "samples": len(references[person_id])}

    def recognize(self, *, face_jpeg: bytes | None = None, pcm16: bytes | None = None,
                  face_captured_at: float | None = None, voice_captured_at: float | None = None) -> dict:
        """Require fresh face and voice agreement before returning a member ID.

        pcm16 is mono 16 kHz signed little-endian PCM. Capture times must come from
        the device/session, not an untrusted image's EXIF. Missing timestamps cannot
        silently convert an old stored photo to a fresh observation.
        """
        evidence, observations = {}, {}
        for modality, data, captured_at in (("face", face_jpeg, face_captured_at),
                                             ("voice", pcm16, voice_captured_at)):
            if data is None:
                evidence[modality] = {"person_id": None, "reason": "missing_input"}
                continue
            if captured_at is None:
                evidence[modality] = {"person_id": None, "reason": "missing_capture_time"}
                continue
            try:
                observation = self._extract(modality, data, captured_at)
                observations[modality] = observation
                evidence[modality] = self.match(observation)
            except (QualityError, ValueError) as exc:
                evidence[modality] = {"person_id": None, "reason": str(exc)}
        result = {"person_id": None, "reason": "insufficient_evidence", "evidence": evidence}
        if any(item["reason"] == "ambiguous_candidates" for item in evidence.values()):
            result["reason"] = "ambiguous_candidates"
            return result
        face, voice = evidence["face"].get("person_id"), evidence["voice"].get("person_id")
        if face and voice and face != voice:
            result["reason"] = "modality_conflict"
        elif face and voice:
            if abs(observations["face"].captured_at - observations["voice"].captured_at) > self.policy.max_modality_gap_seconds:
                result["reason"] = "capture_time_mismatch"
            else:
                result.update(person_id=face, reason="face_voice_agree")
        return result
