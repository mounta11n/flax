from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, unfreeze
from jax import lax
from jax.random import PRNGKey
from transformers import GPT2Config, FlaxViTModel, ViTConfig
from transformers.modeling_flax_outputs import (
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from transformers.models.bart.modeling_flax_bart import (
    shift_tokens_right,
)
from .modeling_flax_gpt2 import (
    FlaxGPT2Module,
    FlaxGPT2Model,
    FlaxGPT2LMHeadModule,
    FlaxGPT2LMHeadModel,
    FlaxPreTrainedModel
)
from transformers.models.vit.modeling_flax_vit import FlaxViTModule

from .configuration_vit_gpt2 import ViTGPT2Config


def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = jnp.roll(input_ids, 1, axis=-1)
    shifted_input_ids = jax.ops.index_update(shifted_input_ids, (..., 0), decoder_start_token_id)
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)

    return shifted_input_ids

class FlaxViTGPT2LMModule(nn.Module):
    config: ViTGPT2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):

        self.encoder = FlaxViTModule(self.config.vit_config, dtype=self.dtype)
        self.decoder = FlaxGPT2LMHeadModule(self.config.gpt2_config, dtype=self.dtype)

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def __call__(
            self,
            pixel_values,
            input_ids,
            attention_mask,
            position_ids,
            encoder_attention_mask: Optional[jnp.ndarray] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            deterministic: bool = True,
    ):
        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqLMOutput(
            logits=decoder_outputs.logits,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class FlaxViTGPT2LMForConditionalGenerationModule(nn.Module):
    config: ViTGPT2Config
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.model = FlaxViTGPT2LMModule(config=self.config, dtype=self.dtype)

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        position_ids,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        return outputs


class FlaxViTGPT2LMPreTrainedModel(FlaxPreTrainedModel):
    config_class = ViTGPT2Config
    base_model_prefix: str = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: ViTGPT2Config,
        input_shape: Tuple = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (
                (1, config.vit_config.image_size, config.vit_config.image_size, 3),
                (1, 1),
            )

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config, module, input_shape=input_shape, seed=seed, dtype=dtype
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jax.random.normal(rng, input_shape[0])
        # # make sure initialization pass will work for FlaxBartForSequenceClassificationModule
        # input_ids = jax.ops.index_update(input_ids, (..., -1), self.config.eos_token_id)

        input_ids = jnp.zeros(input_shape[1], dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        batch_size, sequence_length = input_ids.shape
        position_ids = jnp.broadcast_to(
            jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(
            rngs,
            pixel_values,
            input_ids,
            attention_mask,
            position_ids,
        )["params"]

    def init_cache(self, batch_size, max_length, encoder_outputs):

        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
            input_ids.shape,
        )

        def _decoder_forward(
            module,
            input_ids,
            attention_mask,
            position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids,
                attention_mask,
                position_ids,
                **kwargs,
            )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # we only need to call the decoder to init the cache
        )
        return unfreeze(init_variables["cache"])

    def encode(
        self,
        pixel_values: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, pixel_values, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(pixel_values, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            pixel_values=jnp.array(pixel_values, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    def decode(
        self,
        input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`."
                )

            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxGPT2Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            input_ids,
            attention_mask,
            position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids,
                attention_mask,
                position_ids,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def __call__(
        self,
        pixel_values: jnp.ndarray,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # # prepare encoder inputs
        # if encoder_attention_mask is None:
        #     encoder_attention_mask = jnp.ones_like(input_ids)

        # if position_ids is None:
        #     batch_size, sequence_length = input_ids.shape
        #     position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # prepare decoder inputs
        # if decoder_input_ids is None:
        #     decoder_input_ids = shift_tokens_right(
        #         input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
        #     ) # TODO: Check how to use this

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            pixel_values=jnp.array(pixel_values, dtype=jnp.float32),
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


class FlaxViTGPT2LMForConditionalGeneration(FlaxViTGPT2LMPreTrainedModel):
    module_class = FlaxViTGPT2LMForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    def decode(
        self,
        input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        deterministic: bool = True,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`."
                )

            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxGPT2Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            input_ids,
            attention_mask,
            position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(
                input_ids,
                attention_mask,
                position_ids,
                **kwargs,
            )
            lm_logits = outputs[0]

            return lm_logits, outputs

        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, outputs = outputs
        else:
            (lm_logits, outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=outputs.decoder_hidden_states,
                attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length,
        encoder_attention_mask: Optional[jnp.DeviceArray] = None,
        attention_mask: Optional[jnp.DeviceArray] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": encoder_attention_mask,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = (
            model_kwargs["position_ids"][:, -1:] + 1
        )
        return model_kwargs

    @classmethod
    def from_vit_gpt2_pretrained(
        cls,
        vit_model_name_or_path: str = None,
        gpt2_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxViTGPT2LMPreTrainedModel:

        kwargs_gpt2 = {
            argument[len("gpt2_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("gpt2_")
        }

        kwargs_vit = {
            argument[len("vit_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vit_")
        }

        # remove gpt2, vit kwargs from kwargs
        for key in kwargs_gpt2.keys():
            del kwargs["gpt2_" + key]
        for key in kwargs_vit.keys():
            del kwargs["vit_" + key]

        # Load and initialize the gpt2 and vit model
        gpt2_model = kwargs_gpt2.pop("model", None)
        if gpt2_model is None:
            assert (
                gpt2_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `gpt2_model_name_or_path` has to be defined"

            if "config" not in kwargs_gpt2:
                gpt2_config = GPT2Config.from_pretrained(gpt2_model_name_or_path)
                kwargs_gpt2["config"] = gpt2_config

            kwargs_gpt2["config"].add_cross_attention = True
            gpt2_model = FlaxGPT2LMHeadModel.from_pretrained(
                gpt2_model_name_or_path, *model_args, **kwargs_gpt2
            )

        vit_model = kwargs_vit.pop("model", None)
        if vit_model is None:
            assert (
                vit_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `vit_model_name_or_path` has to be defined"

            if "config" not in kwargs_vit:
                vit_config = ViTConfig.from_pretrained(vit_model_name_or_path)
                kwargs_vit["config"] = vit_config

            vit_model = FlaxViTModel.from_pretrained(
                vit_model_name_or_path, *model_args, **kwargs_vit
            )

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = ViTGPT2Config.from_vit_gpt2_configs(
            vit_model.config, gpt2_model.config, **kwargs
        )

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)
        model.params["model"]["encoder"] = vit_model.params
        model.params["model"]["decoder"] = gpt2_model.params

        return model
