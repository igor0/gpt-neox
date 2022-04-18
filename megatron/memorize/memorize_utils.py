import torch
import tqdm

from typing import List

from megatron.text_generation_utils import forward_model, get_batch
from megatron.utils import print_rank_0

def memorize(
    neox_args,
    model,
    tokens: List[int]
):
    first_batch_end = min(len(tokens), neox_args.seq_length)
    last_batch_end = len(tokens)

    context_tokens = torch.cuda.LongTensor(tokens).unsqueeze(dim=0)

    input_ids, attention_mask, position_ids = get_batch(neox_args, context_tokens[:,:first_batch_end])

    assert input_ids.shape[0] == 1

    with torch.no_grad():
        layer_past = torch.Tensor().cuda()

        for batch_end in tqdm.trange(first_batch_end, last_batch_end+1, desc="Memorizing"):
            model_inputs = (
                input_ids,
                position_ids,
                attention_mask,
                layer_past,
            )

            logits, layer_past = forward_model(neox_args, model, model_inputs)

            generated_token_logits = logits[:, -1].view(1, -1).contiguous()
            generated_tokens = torch.argmax(generated_token_logits, dim=-1).view(-1)

            if batch_end == last_batch_end:
                # This is the last loop iteration. No need to prepare the state the next iteration.
                break

            # XXX - debugging
            #print("###############################")
            #print(neox_args.tokenizer.detokenize(input_ids.tolist()[0])),
            #print(">", neox_args.tokenizer.detokenize(generated_tokens.tolist()))
            #print("###############################")
            # Set input_ids to a single token
            # TODO: support arbitrary stride, not just 1-token stride
            input_ids = input_ids[:, :1]
            input_ids[:, 0] = context_tokens[:, batch_end]

            # Advance layer_past by one token
            # layer_past: [layer, kv, seq, batch, head, dim])
            layer_past = layer_past[:,:,1:,:,:,:]

def memorize_from_file(
    neox_args,
    model,
    input_file,
):
    # Read the sample file
    print_rank_0(
        "memorize_from_file() loading input from {}".format(input_file)
    )
    with open(input_file, "r") as f:
        text = f.read()
        tokens = neox_args.tokenizer.tokenize(text)

    memorize(neox_args, model, tokens)

def _make_batches():
    while True:
        model_inputs = (
            input_ids,
            position_ids,
            attention_mask,
            layer_past,
        )

        logits, layer_past = forward_model(neox_args, model, model_inputs)

        generated_token_logits = logits[:, -1].view(1, -1).contiguous()
        generated_tokens = torch.argmax(generated_token_logits, dim=-1).view(-1)

        # XXX - debugging
        #print("###############################")
        #print(neox_args.tokenizer.detokenize(input_ids.tolist()[0])),
        #print(">", neox_args.tokenizer.detokenize(generated_tokens.tolist()))
        #print("###############################")

        if batch_end == context_tokens.shape[1]:
            break

        # Set input_ids to a single token
        # TODO: support arbitrary stride, not just 1-token stride
        input_ids = input_ids[:, :1]
        input_ids[:, 0] = context_tokens[:, batch_end]

        # Advance layer_past by one token
        # layer_past: [layer, kv, seq, batch, head, dim])
        layer_past = layer_past[:,:,1:,:,:,:]

        batch_end += 1
        if batch_end % 100 == 0:
            print_rank_0("Memorizing...", batch_end, "/", context_tokens.shape[1], end='\r')
