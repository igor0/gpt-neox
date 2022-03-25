from megatron.utils import is_local_main, print_rank_0
import best_download

# patch best_download (eval harness downloader) to only happen on the first local rank
fn = best_download.download_file


def _download_file(*args, **kwargs):
    if is_local_main():
        fn(*args, **kwargs)


best_download.download_file = _download_file

import os
import sys
from functools import partial

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from tqdm import tqdm
import torch
import torch.nn.functional as F

from lm_eval.base import CacheHook
from lm_eval.models.gpt2 import GPT2LM
from lm_eval import tasks, evaluator, utils
from megatron.text_generation_utils import generate_samples_from_prompt
from megatron import mpu


class EvalHarnessAdapter(GPT2LM):
    """
    An adapter to run NeoX models on LM Evaluation Harness (https://github.com/EleutherAI/lm-evaluation-harness) tasks.

    Args:
        model: A NeoX Model
        forward_step_fn: A function that runs a forward pass through the model, returning `tuple(loss, logits)`.
        neox_args: a NeoXArgs object containing the model configuration.
        batch_size (optional): An argument to override the batch size, which defaults to batch size per gpu * dp world size.
    """
    def __init__(self, model, forward_step_fn, neox_args, batch_size=None):

        self.device = torch.device(f"cuda:{neox_args.local_rank}")
        self.VOCAB_SIZE = neox_args.padded_vocab_size
        self.tokenizer = neox_args.tokenizer
        self.EOT_TOKEN_ID = neox_args.tokenizer.eod_id
        self.model = model
        self.neox_args = neox_args
        self.cache_hook = CacheHook(None)
        self.max_length = neox_args.max_position_embeddings // 2
        self.max_gen_toks = 128

        if self.neox_args.pythia_predict_special in ["next2", "prev2"]:
            self.max_length += 1

        # parallelism args:
        self.is_main = neox_args.rank == 0
        self.is_local_main = neox_args.local_rank == 0
        self.is_model_parallel = neox_args.model_parallel_size > 1
        self.is_pipe_parallel = self.model.is_pipe_parallel
        self.is_data_parallel = self.model.is_data_parallel
        self.is_last_stage = (
            True if not self.is_pipe_parallel else model.is_last_stage()
        )  # only the last stage of the pipeline model will receive the logits
        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_group = mpu.get_data_parallel_group()
        self.is_mp_rank_0 = mpu.get_model_parallel_rank() == 0

        self.batch_size = batch_size or (neox_args.batch_size * self.dp_world_size) # default batch size to bs per gpu * dp size

        # some utility functions:
        # we need to patch tokenizer methods, because lm_eval uses them internally:
        self.tokenizer.encode = self.tokenizer.tokenize
        self.tokenizer.decode = self.tokenizer.detokenize
        self._forward_step_fn = partial(
            forward_step_fn, neox_args=neox_args, timers=None, return_logits=True
        )
        self.generate = partial(
            generate_samples_from_prompt,
            neox_args=neox_args,
            model=model,
            maximum_tokens=self.max_gen_toks,
            temperature=0.0,
            broadcast_generated_tokens=True,
        )

    def greedy_until(self, requests):
        """
        Greedy until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.

        TODO: batched / data parallel generation

        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """
        self.model.module.inference_mode(cache=True)  # tell model to cache kv pairs
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(requests, _collate)
        for context, until in tqdm(reord.get_reordered(), "Running greedy generation"):
            if isinstance(until, str):
                until = [until]
            stop_tokens = [self.tokenizer.encode(i) for i in until]
            cont = self.generate(
                text=context,
                stop_tokens=stop_tokens,
                recompute=self.neox_args.recompute,
            )
            if cont:
                s = cont[0]["text"] or ""
            else:
                s = ""

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        self.model.module.train_mode()  # set back to train mode
        return reord.get_original(res)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """
        In this method, the model doesn't do any generation, but just returns log likelihoods
        for the next token, which eval harness uses to evaluate.

        :param requests: Dictionary of requests containing the context and the expected continuation.
        :param disable_tqdm: If True, disable tqdm progress bar.
        """
        self.model.module.inference_mode(
            cache=False
        )  # tell model to gather parallel outputs, but not cache key-value pairs

        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        with torch.no_grad():

            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(
                tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size
            ):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    if self.neox_args.pythia_predict_special == None:
                        inp_tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
                    elif self.neox_args.pythia_predict_special == "next":
                        inp_tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
                    elif self.neox_args.pythia_predict_special == "next2":
                        # here, max_length is already +1 (see __init__)
                        inp_tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
                    else:
                        # self or prev or or prev2 abslog
                        assert self.neox_args.pythia_predict_special in ["self", "prev", "prev2", "abslog"]

                        # Note: prev and self are handled the same because the logits will already be
                        # shifted appropriately. next required the adjustment shown above.
                        inp_tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :][1:]

                    inp = torch.tensor(
                        inp_tokens,
                        dtype=torch.long,
                    ).to(self.device)
                    (inplen,) = inp.shape

                    cont = continuation_enc
                    if self.neox_args.pythia_predict_special == "abslog":
                        cont = torch.arange(1, len(cont) + 1).log2().floor().long()

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = (
                        padding_length if padding_length is not None else inplen
                    )

                    # pad to length
                    inp = torch.cat(
                        [
                            inp,  # [seq]
                            torch.zeros(padding_length - inplen, dtype=torch.long).to(
                                inp.device
                            ),  # [padding_length - seq]
                        ],
                        dim=0,
                    )

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)

                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                        chunk, multi_logits, inps, inplens, contlens
                    ):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen : inplen].unsqueeze(
                            0
                        )  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = (
                            torch.tensor(cont_toks, dtype=torch.long)
                            .unsqueeze(0)
                            .to(multi_logits.device)
                        )

                        if self.neox_args.pythia_predict_special == "next2":
                            cont_toks = cont_toks[:,1:]
                        elif self.neox_args.pythia_predict_special == "prev2":
                            cont_toks = cont_toks[:,:-1]
                        max_equal = (greedy_tokens == cont_toks).all()
                        logits = torch.gather(
                            logits, 2, cont_toks.unsqueeze(-1)
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))

                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer
                            )

                        res.append(answer)

            # broadcast results to all ranks
            if self.is_pipe_parallel:
                src_rank = self.model.grid.stage_to_global(self.model.num_stages - 1)
                if res:
                    logits_sums, max_equals = list(zip(*res))
                    logits_sums = torch.FloatTensor(logits_sums).cuda()
                    max_equals = torch.LongTensor(max_equals).cuda()
                else:
                    logits_sums = torch.zeros(res_len, dtype=torch.float32).cuda()
                    max_equals = torch.zeros(res_len, dtype=torch.int64).cuda()
                torch.distributed.broadcast(tensor=logits_sums, src=src_rank)
                torch.distributed.broadcast(tensor=max_equals, src=src_rank)
                max_equals = [bool(i) for i in max_equals.tolist()]
                logits_sums = logits_sums.tolist()
                res = list(zip(logits_sums, max_equals))

        self.model.module.train_mode()  # set back to train mode
        return reord.get_original(res)

    def _dp_scatter(self, inps):
        """
        Scatters the inputs to all data parallel ranks.
        """

        batch_size = inps.shape[0]
        padded = False
        if batch_size % self.dp_world_size != 0:
            # The last batch could potentially not fill the full batch size (if the dataset size is not divisible by batch size)
            # In this case we pad the batch
            padded_size = self.dp_world_size - (batch_size % self.dp_world_size)

            print_rank_0(f'WARNING: Batch size ({batch_size}) must be divisible by dp world size ({self.dp_world_size}). Padding inputs to {padded_size}.')
            
            inps = torch.cat([inps] + [ inps[0:1, ...] for _ in range(padded_size) ], dim=0) # pad with first inp item
            padded = True

        assert (
            inps.shape[0] % self.dp_world_size == 0
        ), f"batch size ({inps.shape[0]}) must be divisible by dp world size ({self.dp_world_size})"

        # get a chunk for each data parallel rank
        chunk_size = inps.shape[0] // self.dp_world_size
        inps = inps[self.dp_rank * chunk_size : (self.dp_rank + 1) * chunk_size]
        # make a dummy dataloader / iterator to pass to model
        # we need to do this because deepspeed pipe parallel only takes an iterator
        # in this format
        return iter([{"text": F.pad(inps, pad=(0, 1))}]), padded

    def _dp_gather(self, logits):
        """
        Gather logits from all data parallel ranks
        """
        if logits is not None:
            tensor_list = [torch.zeros_like(logits) for _ in range(self.dp_world_size)]
            torch.distributed.all_gather(
                tensor_list, logits, group=mpu.get_data_parallel_group()
            )
            logits = torch.cat(tensor_list, dim=0)
            return logits 

    def _model_call(self, inps):
        batch_size = inps.shape[0]

        # scatter inputs to all dp ranks:
        inps, padded = self._dp_scatter(inps)

        if self.neox_args.is_pipe_parallel:
            # need these flags to stop deepspeed pipe parallel from hanging
            self.model.first_output_send = True
            self.model.pipe_recv_buf = None

        _, logits = self._forward_step_fn(model=self.model, data_iterator=inps)

        # gather outputs from all dp ranks:
        logits = self._dp_gather(logits)

        # if logits have been padded (normally just last item where batch size is unequal)
        # restore to original shape
        if padded and logits is not None:
            logits = logits[:batch_size, ...]
        return logits

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0):
        was_training = self.model.training
        self.model.eval()
        in_micro_batches = (
            self.model.micro_batches
        )  # store input microbatches - we need to set to 1 during eval, but want to return to its original value after
        self.model.micro_batches = 1
        if eval_tasks is None:
            eval_tasks = [
                "lambada",
                "piqa",
                "hellaswag",
                "winogrande",
                "mathqa",
                "pubmedqa",
            ]

        # **HACK INCOMING**:
        # first get task dict on local main rank
        # the tasks are downloaded *as they are initialized*, and the downloads don't like multithreading.
        # so we download them once on the local main rank, wait, and then initialize them on all other ranks, which *should* load from the cache.
        if self.is_local_main:
            task_dict = tasks.get_task_dict(eval_tasks)
        # torch barrier
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        task_dict = tasks.get_task_dict(eval_tasks)

        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=num_fewshot,
            limit=None,
            bootstrap_iters=2,
        ).get("results")

        if was_training:
            self.model.train()
        self.model.micro_batches = in_micro_batches
        return results


def run_eval_harness(
    model, forward_step_fn, neox_args, batch_size=None, eval_tasks=None, num_fewshot=0,
):
    print_rank_0("Running evaluation harness...")
    adapter = EvalHarnessAdapter(model, forward_step_fn, neox_args, batch_size)
    return adapter.run_eval(eval_tasks=eval_tasks, num_fewshot=num_fewshot)
