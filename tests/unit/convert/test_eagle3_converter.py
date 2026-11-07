"""
Unit tests for Eagle3 converter focusing on the specific fixes implemented.

Tests cover:
- Embeddings replacement with verifier embeddings
- Weight remapping from midlayer.* to layers.0.*
- Configuration compatibility (max_position_embeddings, rope_theta)
- Validation of the conversion process
"""

import logging
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.utils import load_checkpoint_config

# A tiny but complete Eagle-3 source checkpoint: small enough to build in-process,
# shaped so every Eagle3Speculator parameter has a supplier.
_H, _I, _V, _DV = 64, 128, 256, 32
_HEADS, _KV_HEADS = 4, 2
_KV_DIM = (_H // _HEADS) * _KV_HEADS

_TINY_EAGLE_CONFIG = {
    "target_vocab_size": _V,
    "hidden_size": _H,
    "intermediate_size": _I,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "hidden_act": "silu",
    "max_position_embeddings": 128,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "rope_theta": 10000.0,
    "mlp_bias": False,
    "draft_vocab_size": _DV,
}

_TINY_VERIFIER_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "max_position_embeddings": 512,
    "rope_theta": 500000.0,
    "vocab_size": _V,
}


def _tiny_eagle3_weights() -> dict[str, torch.Tensor]:
    """Source-side weights, in the legacy ``midlayer.*`` naming the converter remaps."""
    return {
        "embed_tokens.weight": torch.randn(_V, _H),
        "fc.weight": torch.randn(_H, 3 * _H),
        # q/k/v read the embedding concatenated with the projected hidden state,
        # so their input dimension is 2 * hidden_size.
        "midlayer.self_attn.q_proj.weight": torch.randn(_H, 2 * _H),
        "midlayer.self_attn.k_proj.weight": torch.randn(_KV_DIM, 2 * _H),
        "midlayer.self_attn.v_proj.weight": torch.randn(_KV_DIM, 2 * _H),
        "midlayer.self_attn.o_proj.weight": torch.randn(_H, _H),
        "midlayer.mlp.gate_proj.weight": torch.randn(_I, _H),
        "midlayer.mlp.up_proj.weight": torch.randn(_I, _H),
        "midlayer.mlp.down_proj.weight": torch.randn(_H, _I),
        "midlayer.input_layernorm.weight": torch.randn(_H),
        "midlayer.hidden_norm.weight": torch.randn(_H),
        "midlayer.post_attention_layernorm.weight": torch.randn(_H),
        "norm.weight": torch.randn(_H),
        "lm_head.weight": torch.randn(_DV, _H),
        "t2d": torch.zeros(_V, dtype=torch.bool),
        "d2t": torch.zeros(_DV, dtype=torch.long),
    }


def _save_tiny_checkpoint(weights: dict[str, torch.Tensor], output_dir):
    """Run ``_save_converted_checkpoint`` for the tiny checkpoint above."""
    converter = Eagle3Converter()
    with patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict",
        return_value=(_TINY_VERIFIER_CONFIG, None),
    ):
        config = converter._build_eagle3_speculator_config(
            dict(_TINY_EAGLE_CONFIG), "base/model"
        )
        return converter._save_converted_checkpoint(
            config,
            weights,
            output_dir,
            reduce_vocab_size="t2d" in weights,
            has_drafter_embedding="embed_tokens.weight" in weights,
        )


class TestEagle3ConverterKeyMismatch:
    """A key mismatch must fail the conversion, not warn and save noise."""

    @pytest.mark.sanity
    def test_complete_checkpoint_converts_and_preserves_weights(self, tmp_path):
        """The baseline: a complete checkpoint still converts, with real weights."""
        weights = _tiny_eagle3_weights()
        saved = _save_tiny_checkpoint(weights, tmp_path / "out")

        written = load_file(saved / "model.safetensors")
        # midlayer.* landed as layers.0.*, carrying the source values.
        assert torch.allclose(
            written["layers.0.self_attn.q_proj.weight"],
            weights["midlayer.self_attn.q_proj.weight"],
        )
        assert torch.allclose(written["norm.weight"], weights["norm.weight"])

    @pytest.mark.sanity
    def test_renamed_weight_key_raises(self, tmp_path):
        """The shape of the #360 regression: a lost remap must not save silently.

        Renaming a key leaves the destination parameter unsupplied, so the
        missing-key check is what catches it.
        """
        weights = _tiny_eagle3_weights()
        weights["midlayer.input_layernorm_RENAMED.weight"] = weights.pop(
            "midlayer.input_layernorm.weight"
        )

        with pytest.raises(ValueError, match="random initialization") as excinfo:
            _save_tiny_checkpoint(weights, tmp_path / "out")

        assert "layers.0.input_layernorm.weight" in str(excinfo.value)
        assert not (tmp_path / "out" / "model.safetensors").exists()

    @pytest.mark.sanity
    def test_extra_layer_weights_warn_but_still_convert(self, tmp_path, caplog):
        """Eagle3Speculator builds one layer, so extra layers have no destination.

        Those weights are dropped rather than written, which the caller needs to
        be told about -- but it is not the random-weight hazard, so it is not fatal.
        """
        weights = _tiny_eagle3_weights()
        weights["midlayer_1.self_attn.q_proj.weight"] = torch.randn(_H, _H)

        with caplog.at_level(logging.WARNING):
            saved = _save_tiny_checkpoint(weights, tmp_path / "out")

        written = load_file(saved / "model.safetensors")
        assert "midlayer_1.self_attn.q_proj.weight" not in written

    @pytest.mark.sanity
    def test_missing_weight_key_raises(self, tmp_path):
        """A parameter with no supplier would be written at random init."""
        weights = _tiny_eagle3_weights()
        del weights["midlayer.mlp.down_proj.weight"]

        with pytest.raises(ValueError, match="random initialization"):
            _save_tiny_checkpoint(weights, tmp_path / "out")

        assert not (tmp_path / "out" / "model.safetensors").exists()

    @pytest.mark.sanity
    def test_absent_d2t_buffer_still_converts(self, tmp_path):
        """d2t is zero-initialized, not random, so its absence is not fatal."""
        weights = _tiny_eagle3_weights()
        del weights["d2t"]

        saved = _save_tiny_checkpoint(weights, tmp_path / "out")

        written = load_file(saved / "model.safetensors")
        assert torch.equal(written["d2t"], torch.zeros(_DV, dtype=torch.long))

    @pytest.mark.sanity
    def test_checkpoint_without_vocab_mapping_converts(self, tmp_path):
        """No t2d means no vocab-map buffers are registered at all."""
        weights = _tiny_eagle3_weights()
        del weights["t2d"]
        del weights["d2t"]

        saved = _save_tiny_checkpoint(weights, tmp_path / "out")

        written = load_file(saved / "model.safetensors")
        assert "t2d" not in written
        assert "d2t" not in written


class TestEagle3ConverterFixes:
    """Test the specific fixes implemented in Eagle3 converter."""

    @pytest.fixture
    def sample_eagle3_config(self):
        """Sample Eagle3 configuration for testing."""
        return {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 2048,  # This will be maxed with verifier
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 10000.0,
            "mlp_bias": False,
        }

    @pytest.fixture
    def sample_eagle3_weights(self):
        """Sample Eagle3 weights with midlayer.* naming."""
        return {
            # Embeddings that should be replaced
            "embed_tokens.weight": torch.randn(128000, 4096),
            # Weights that should be remapped from midlayer.* to layers.0.*
            "midlayer.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "midlayer.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "midlayer.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "midlayer.self_attn.o_proj.weight": torch.randn(4096, 4096),
            "midlayer.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "midlayer.mlp.up_proj.weight": torch.randn(11008, 4096),
            "midlayer.mlp.down_proj.weight": torch.randn(4096, 11008),
            "midlayer.input_layernorm.weight": torch.randn(4096),
            "midlayer.post_attention_layernorm.weight": torch.randn(4096),
            # Other weights
            "t2d": torch.randn(128000, 4096),
        }

    @pytest.fixture
    def sample_verifier_config(self):
        """Sample verifier configuration."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 131072,  # Should be maxed with Eagle3's 2048
            "rope_theta": 500000.0,
            "vocab_size": 128256,
        }

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_max_position_embeddings_logic(
        self, mock_get_config, sample_eagle3_config, sample_verifier_config
    ):
        """Test that max_position_embeddings uses max of Eagle3 and verifier."""
        mock_get_config.return_value = (sample_verifier_config, None)

        converter = Eagle3Converter()

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )

        # Check that max_position_embeddings is the max of both values
        # Eagle3: 2048, Verifier: 131072, so should be 131072
        assert llama_config.max_position_embeddings == 131072

        # Check that other values come from Eagle3 config
        assert llama_config.hidden_size == 4096
        assert llama_config.num_attention_heads == 32
        # rope_theta comes from Eagle3 config, not verifier
        if hasattr(llama_config, "rope_parameters"):
            # Transformers v5
            assert llama_config.rope_parameters is not None
            assert llama_config.rope_parameters.get("rope_theta") == 10000.0
        else:
            assert llama_config.rope_theta == 10000.0

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_num_hidden_layers_from_config(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test that num_hidden_layers is taken from eagle_config when present."""
        mock_get_config.return_value = ({}, None)
        converter = Eagle3Converter()

        # Add num_hidden_layers to the sample config
        sample_eagle3_config["num_hidden_layers"] = 3

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert llama_config.num_hidden_layers == 3

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_num_hidden_layers_default(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test that num_hidden_layers defaults to 1 when not in config."""
        mock_get_config.return_value = ({}, None)
        converter = Eagle3Converter()

        # Remove num_hidden_layers if present
        sample_eagle3_config.pop("num_hidden_layers", None)

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert llama_config.num_hidden_layers == 1

    @pytest.mark.sanity
    @pytest.mark.parametrize("dtype_key", ["torch_dtype", "dtype"])
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_dtype_from_either_key(
        self, mock_get_config, dtype_key, sample_eagle3_config
    ):
        """Checkpoints saved by newer transformers write ``dtype``, older ones
        ``torch_dtype``; both must set the dtype the weights are saved in."""
        mock_get_config.return_value = ({}, None)
        sample_eagle3_config[dtype_key] = "bfloat16"

        llama_config = Eagle3Converter()._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        # The dtype _save_converted_checkpoint casts the weights to.
        saved_dtype = getattr(llama_config, "dtype", None) or getattr(
            llama_config, "torch_dtype", None
        )
        assert saved_dtype == torch.bfloat16

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_fallback_when_verifier_unavailable(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test fallback behavior when verifier config cannot be loaded."""
        mock_get_config.side_effect = Exception("Network error")

        converter = Eagle3Converter()

        # Should raise exception since method doesn't handle this gracefully
        with pytest.raises((RuntimeError, ValueError)):
            converter._create_transformer_config_from_eagle(
                sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
            )

    @pytest.mark.sanity
    def test_converted_model_config_structure(self):
        """Test that the config structure created is valid for Eagle3Speculator."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            config = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
            )

        # Verify config structure
        assert config.speculators_model_type == "eagle3"
        assert config.transformer_layer_config.hidden_size == 4096
        assert config.speculators_config.algorithm == "eagle3"
        assert (
            config.speculators_config.verifier.name_or_path
            == "meta-llama/Llama-3.1-8B-Instruct"
        )

    @pytest.mark.sanity
    def test_eagle_aux_hidden_state_layer_ids_parameter(self):
        """Test that eagle_aux_hidden_state_layer_ids parameter is properly handled."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            # Test with None (default)
            config_none = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
            )
            assert config_none.eagle_aux_hidden_state_layer_ids is None

            # Test with specific layer IDs
            layer_ids = [1, 23, 44]
            config_with_ids = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
                eagle_aux_hidden_state_layer_ids=layer_ids,
            )
            assert config_with_ids.eagle_aux_hidden_state_layer_ids == layer_ids

    @pytest.mark.sanity
    def test_eagle_aux_hidden_state_layer_ids_in_config_serialization(self):
        """Test that eagle_aux_hidden_state_layer_ids is properly serialized."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            layer_ids = [1, 23, 44]
            config = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
                eagle_aux_hidden_state_layer_ids=layer_ids,
            )

            # Test that the config can be serialized to dict
            config_dict = config.to_dict()
            assert "eagle_aux_hidden_state_layer_ids" in config_dict
            assert config_dict["eagle_aux_hidden_state_layer_ids"] == layer_ids

    @pytest.mark.sanity
    def test_nm_testing_2layer_eagle3_model_config(self, tmp_path):
        """Test that multi-layer conversion support."""
        converter = Eagle3Converter()

        # TODO: Use a real model in the future
        checkpoint_path = "nm-testing/testing-llama3.1.8b-2layer-eagle3"
        base_model = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"

        local_checkpoint_path = tmp_path / checkpoint_path.rsplit("/", maxsplit=1)[-1]
        converter.convert(
            checkpoint_path,
            local_checkpoint_path,
            base_model,
            norm_before_residual=False,
        )

        config = load_checkpoint_config(local_checkpoint_path)

        # Verify that num_hidden_layers is correctly set to 2
        assert config["transformer_layer_config"]["num_hidden_layers"] == 2
