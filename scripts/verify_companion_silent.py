"""Run the explicitly selected silent suite with the companion Python environment.

No discovery: audio, real voiceprint inference, TTS and hardware checks are
deliberately excluded. Identity API tests use fake engines and temporary data.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = (
    "test_server", "test_learning", "test_learning_gateway", "test_identity_api",
    "test_patrol", "test_remote", "test_remote_runtime", "test_agent",
    "test_memory", "test_story", "test_story_quality", "test_inference",
    "test_launch", "test_custom_action_chat",
    "test_identity_settings", "test_camera_dialogue", "test_cancelled_memory",
    "test_gateway_cancel", "test_gesture",
    "test_offline_library", "test_gateway_config", "test_tls", "test_routine",
    "test_action_drafts", "test_pet_workflows", "test_panel_assets", "test_model_response_api",
    "test_model_generation", "test_draft_history", "test_reply_boundary", "test_learning_commit",
    "test_member_revocation", "test_personal_access", "test_english_questions",
    "test_operation_replies", "test_rename_proposal", "test_memory_statements", "test_verbatim_agent",
)

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ["companion." + name for name in MODULES])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
