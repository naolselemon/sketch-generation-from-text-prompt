from __future__ import annotations

import unittest

from scripts.sketchformer.config import compose_training_config


class ConfigCompositionTest(unittest.TestCase):
    def test_default_training_config_targets_tok_dict(self) -> None:
        config = compose_training_config("configs/train.yaml", experiment="smoke_test")

        self.assertEqual(config["data"]["format"]["type"], "tok_dict")
        self.assertEqual(config["model"]["name"], "sketchformer_tok_dict")
        self.assertEqual(
            config["model"]["input"]["token_dictionary"]["pad_token_id"],
            config["data"]["format"]["token_dictionary"]["pad_token_id"],
        )
        self.assertEqual(config["trainer"]["checkpointing"]["monitor"], "val/token_loss")


if __name__ == "__main__":
    unittest.main()
