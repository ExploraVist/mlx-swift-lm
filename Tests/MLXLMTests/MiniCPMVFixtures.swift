// Real config files used as fixtures below, fetched verbatim from Hugging Face:
//   openbmb/MiniCPM-V-4.6                 config.json, preprocessor_config.json
//   mlx-community/MiniCPM-V-4.6-4bit      config.json
//   mlx-community/MiniCPM-V-4.7-1B-4bit   config.json, preprocessor_config.json, processor_config.json

import Foundation

enum MiniCPMVFixtures {
    static let openbmbV46Config = #"""
        {
          "architectures": [
            "MiniCPMV4_6ForConditionalGeneration"
          ],
          "bos_token_id": null,
          "drop_vision_last_layer": false,
          "eos_token_id": 248044,
          "image_size": 1120,
          "model_type": "minicpmv4_6",
          "pad_token_id": null,
          "tie_word_embeddings": true,
          "transformers_version": "5.7.0",
          "use_cache": true,
          "vision_config": {
            "attention_dropout": 0.0,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "image_size": 980,
            "intermediate_size": 4304,
            "layer_norm_eps": 1e-06,
            "model_type": "minicpmv4_6_vision",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 27,
            "patch_size": 14
          },
          "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attn_output_gate": true,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 3584,
            "layer_types": [
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
              "partial_rotary_factor": 0.25,
              "rope_theta": 10000000,
              "rope_type": "default"
            },
            "vocab_size": 248094,
            "model_type": "qwen3_5_text",
            "tie_word_embeddings": true
          },
          "insert_layer_id": 6,
          "image_token_id": 248056,
          "video_token_id": 248057
        }
        """#

    static let mlxV46Config = #"""
        {
            "architectures": [
                "MiniCPMV4_6ForConditionalGeneration"
            ],
            "bos_token_id": null,
            "drop_vision_last_layer": false,
            "eos_token_id": [
                248044,
                248046
            ],
            "image_size": 1120,
            "image_token_id": 248056,
            "insert_layer_id": 6,
            "model_type": "minicpmv4_6",
            "pad_token_id": null,
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            },
            "quantization_config": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            },
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 3584,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "mlp_only_layers": [],
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "vocab_size": 248094,
                "model_type": "qwen3_5_text",
                "tie_word_embeddings": true
            },
            "tie_word_embeddings": true,
            "transformers_version": "5.7.0",
            "use_cache": true,
            "video_token_id": 248057,
            "vision_config": {
                "attention_dropout": 0.0,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "image_size": 980,
                "intermediate_size": 4304,
                "layer_norm_eps": 1e-06,
                "model_type": "minicpmv4_6_vision",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 27,
                "patch_size": 14
            }
        }
        """#

    static let mlxV47Config = #"""
        {
            "architectures": [
                "MiniCPMV4_7ForConditionalGeneration"
            ],
            "downsample_mode": "16x",
            "dtype": "bfloat16",
            "eos_token_id": [
                248044,
                248046
            ],
            "generation_config": {
                "eos_token_id": [
                    248044,
                    248046
                ],
                "transformers_version": "5.2.0",
                "use_cache": true,
                "bos_token_id": 248045
            },
            "image_end_id": 248079,
            "image_start_id": 248078,
            "image_token_id": 248056,
            "insert_layer_id": 6,
            "model_type": "minicpmv4_7",
            "mrope_mode": "canvas",
            "newline_id": 198,
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            },
            "quantization_config": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            },
            "slice_end_id": 248089,
            "slice_start_id": 248088,
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 3584,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "mlp_only_layers": [],
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [
                        11,
                        11,
                        10
                    ],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "vocab_size": 248094,
                "model_type": "qwen3_5_text",
                "tie_word_embeddings": false
            },
            "tie_word_embeddings": false,
            "transformers_version": "5.18.0",
            "use_cache": true,
            "version": 4.7,
            "video_token_id": 248057,
            "vision_config": {
                "attention_dropout": 0.0,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "image_size": 980,
                "intermediate_size": 4304,
                "layer_norm_eps": 1e-06,
                "model_type": "minicpmv4_7_vision",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 27,
                "patch_size": 14
            }
        }
        """#

    static let openbmbV46Preprocessor = #"""
        {
          "image_processor_type": "MiniCPMV4_6ImageProcessor",
          "processor_class": "MiniCPMV4_6Processor",
          "max_slice_nums": 9,
          "scale_resolution": 448,
          "patch_size": 14,
          "use_image_id": true,
          "slice_mode": true,
          "image_mean": [
            0.5,
            0.5,
            0.5
          ],
          "image_std": [
            0.5,
            0.5,
            0.5
          ]
        }
        """#

    static let mlxV47Preprocessor = #"""
        {
          "max_slice_nums": 9,
          "image_mean": [
            0.5,
            0.5,
            0.5
          ],
          "image_std": [
            0.5,
            0.5,
            0.5
          ],
          "mrope_mode": "canvas"
        }
        """#

    static let mlxV47Processor = #"""
        {
          "processor_class": "MiniCPMV4_7Processor",
          "image_processor": {
            "max_slice_nums": 9,
            "patch_size": 14,
            "scale_resolution": 448,
            "slice_mode": true,
            "use_image_id": true,
            "downsample_mode": "16x",
            "pack_image_by_patch": true,
            "image_mean": [
              0.5,
              0.5,
              0.5
            ],
            "image_std": [
              0.5,
              0.5,
              0.5
            ]
          },
          "video_processor": {
            "max_slice_nums": 9,
            "patch_size": 14,
            "scale_resolution": 448,
            "slice_mode": true,
            "use_image_id": true,
            "downsample_mode": "16x",
            "pack_image_by_patch": true,
            "max_num_frames": 128,
            "stack_frames": 1,
            "image_mean": [
              0.5,
              0.5,
              0.5
            ],
            "image_std": [
              0.5,
              0.5,
              0.5
            ]
          }
        }
        """#

}
