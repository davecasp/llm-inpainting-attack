import logging
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class InpaintingWrapper(ABC):
    def __init__(self, device: str, batch_size: int, attack_input_size: int):
        """Base class for diffusion-based inpainting wrappers.

        **Arguments:**

        - `device`: PyTorch device string, e.g. `"cuda"` or `"cpu"`.
        - `batch_size`: Number of sequences to process per forward pass.
        - `attack_input_size`: Number of attack tokens to prepend to each target sequence.
        """
        self.device = device
        self.batch_size = batch_size
        self.attacker_input_size = attack_input_size

    @abstractmethod
    def batched_inpainting(self, targets_list: list[str]) -> list[str]:
        """Generate inpainted attack prompts for a list of target strings.

        **Arguments:**

        - `targets_list`: Target response strings to inpaint against.

        **Returns:**

        A list of inpainted prompt strings, one per input target.
        """
        ...


@dataclass
class MaskingResult:
    masked_ids: Int[Tensor, "batch inpaint_seq"]
    mask_positions: Bool[Tensor, "batch inpaint_seq"]


class LLaDaWrapper(InpaintingWrapper):
    def __init__(
        self,
        device: str,
        model_path: str,
        batch_size: int,
        attack_input_size: int,
        num_diffusion_steps: int,
        temperature: float,
        mask_padding: bool,
    ):
        """Inpainting wrapper for the LLaDA-8B-Base diffusion language model.

        **Arguments:**

        - `device`: PyTorch device string, e.g. `"cuda"` or `"cpu"`.
        - `model_path`: Path to the diffusion model.
        - `batch_size`: Number of sequences to process per forward pass.
        - `attack_input_size`: Number of attack tokens to prepend to each target sequence.
        - `num_diffusion_steps`: Number of denoising steps during generation.
        - `temperature`: Gumbel noise temperature; `0.0` disables noise (greedy).
        - `mask_padding`: Whether to mask padding tokens in the attention mask.
        """
        if num_diffusion_steps > attack_input_size:
            raise ValueError(
                f"num_diffusion_steps ({num_diffusion_steps}) must be <= attack_input_size ({attack_input_size})"
            )

        super().__init__(device=device, batch_size=batch_size, attack_input_size=attack_input_size)

        self.mask_padding = mask_padding
        self.steps = num_diffusion_steps
        self.temperature = temperature

        # set up model
        logger.info("Loading diffusion model...")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype="bfloat16").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.mask_token is None:
            print(self.tokenizer.mask_token)
            self.tokenizer.mask_token = "<|mdm_mask|>"
            print(self.tokenizer.mask_token)

        self.padding_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        print(f"mask_token_id: derived={self.mask_token_id}, hardcoded_default=126336, or 156895")

        # suppressed during generation so the model never predicts structural tokens
        self.special_token_ids = list(
            set(
                int(s)
                for s in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
                if s is not None
            )
        )

    def _encode(self, texts: list[str]) -> Int[Tensor, "batch target_seq"]:
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=True)
        return encoded["input_ids"].to(self.device)

    def _mask_tokens(self, token_ids: Int[Tensor, "batch target_seq"]) -> MaskingResult:
        B, _ = token_ids.shape
        device = token_ids.device

        K = self.attacker_input_size

        # prepend K mask tokens; only this prefix is ever unmasked during inpainting
        mask_block = token_ids.new_full((B, K), self.mask_token_id)
        token_ids = torch.cat([mask_block, token_ids], dim=1)

        mask_positions = torch.zeros(token_ids.shape, dtype=torch.bool, device=device)
        mask_positions[:, :K] = True

        return MaskingResult(masked_ids=token_ids, mask_positions=mask_positions)

    @torch.inference_mode()
    def _add_gumbel_noise(
        self, logits: Float[Tensor, "batch inpaint_seq vocab"], temperature: float
    ) -> Float[Tensor, "batch inpaint_seq vocab"]:
        if temperature == 0:
            return logits

        eps = 1e-20
        return (logits - torch.log(-torch.log(torch.rand_like(logits, dtype=torch.bfloat16) + eps) + eps)) / temperature

    @torch.inference_mode()
    def _forward_process_batched(
        self, batch: Int[Tensor, "batch inpaint_seq"], fixed_mask: Bool[Tensor, "batch inpaint_seq"]
    ) -> Int[Tensor, "batch inpaint_seq"]:
        b, target_len = batch.shape
        device = batch.device

        x = torch.randint(1, target_len + 1, (b,), device=device)

        indices = torch.arange(target_len, device=device).unsqueeze(0).expand(b, -1)
        is_mask = indices < x.unsqueeze(1)

        randperm = torch.argsort(torch.rand(b, target_len, device=device), dim=1)
        is_mask = torch.gather(is_mask, 1, randperm)
        is_mask = is_mask & fixed_mask

        return torch.where(is_mask, self.mask_token_id, batch)

    def _get_num_transfer_tokens(
        self, 
        mask_index: Bool[Tensor, "batch inpaint_seq"], 
        steps: int
    ) -> Int[Tensor, "batch steps"]:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    @torch.inference_mode()
    def _predict_masked(
        self,
        masked_ids: Int[Tensor, "batch inpaint_seq"]
    ) -> Int[Tensor, "batch inpaint_seq"]:
        if self.model.training:
            raise RuntimeError("predict_masked must be run in evaluation mode.")

        x = masked_ids.to(self.device)
        batch_size, _ = x.shape

        known_mask = x != self.mask_token_id
        known_tokens = x.clone()

        num_transfer_tokens = self._get_num_transfer_tokens(~known_mask, self.steps)

        neg_inf = torch.tensor(-float("inf"), device=self.device)

        for s in range(self.steps):
            mask_index = x == self.mask_token_id

            x_l = self._forward_process_batched(x, known_mask)

            attention_mask = None
            if self.mask_padding:
                attention_mask = (x_l != self.padding_token_id).long()

            logits = self.model(x_l, attention_mask=attention_mask).logits
            del x_l
            logits_with_noise = self._add_gumbel_noise(logits, temperature=self.temperature)

            if len(self.special_token_ids) > 0:
                logits_with_noise[:, :, self.special_token_ids] = -float("inf")
            x0 = torch.argmax(logits_with_noise, dim=-1)
            del logits_with_noise

            x0_logits = torch.gather(logits, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).float()
            log_z = torch.logsumexp(logits, dim=-1).float()
            x0_p = (x0_logits - log_z).exp()
            del logits, x0_logits, log_z

            x0 = torch.where(known_mask, known_tokens, x0)

            # only currently masked positions are eligible to be committed
            confidence = torch.where(mask_index, x0_p, neg_inf)
            confidence = torch.where(known_mask, neg_inf, confidence)
            del x0_p, mask_index

            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=self.device)

            ks = torch.clamp(num_transfer_tokens[:, s], min=1)
            k_max = int(ks.max().item())
            _, top_indices = torch.topk(confidence, k=k_max, dim=1)
            rank = torch.arange(k_max, device=self.device).unsqueeze(0)
            valid = rank < ks.unsqueeze(1)
            batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, k_max)
            transfer_index[batch_idx[valid], top_indices[valid]] = True
            del confidence, ks, top_indices, rank, valid, batch_idx

            x[transfer_index] = x0[transfer_index]

        return x

    @torch.inference_mode()
    def batched_inpainting(self, targets_list: list[str]) -> list[str]:
        self.model.eval()

        inpainted_prompts = []
        for index in tqdm(
            range(0, len(targets_list), self.batch_size),
            total=(len(targets_list) + self.batch_size - 1) // self.batch_size,
            desc="Generating attack prompts",
        ):
            batch_targets = targets_list[index : index + self.batch_size]

            target_sequences = [f"\nAnswer: {t}" for t in batch_targets]
            target_ids = self._encode(target_sequences)

            masking_result = self._mask_tokens(target_ids)

            inpainted_prompt_ids_full = self._predict_masked(masked_ids=masking_result.masked_ids)

            attacking_prompt_ids = inpainted_prompt_ids_full[masking_result.mask_positions].view(
                inpainted_prompt_ids_full.size(0), -1
            )

            attacking_prompt_texts = self.tokenizer.batch_decode(attacking_prompt_ids, skip_special_tokens=False)
            generated_prompts = [repr(p) for p in attacking_prompt_texts]
            inpainted_prompts.extend(generated_prompts)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return inpainted_prompts
