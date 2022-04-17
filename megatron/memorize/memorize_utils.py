import torch

from megatron.text_generation_utils import forward_model, get_batch

def memorize(
    neox_args,
    model,
    context_tokens: List[int],
):
    batch_end = min(len(context_tokens), neox_args.seq_length)
    input_ids, attention_mask, position_ids = get_batch(neox_args, context_tokens[:batch_end])

    assert input_ids.shape[0] == 1

    with torch.no_grad():
        layer_past = torch.Tensor().cuda()

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

            print("XXX", neox_args.tokenizer.detokenize(input_ids), "->", neox_args.tokenizer.detokenize(generated_tokens))

            if batch_end == len(context_tokens):
                break

            input_ids = input_ids[:, :1]
            input_ids[0, 0] = context_tokens[batch_end]

            attention_mask = attention_mask[:, :1]
            assert attention_mask[:, 0] == 1

            batch_end += 1

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
        tokens = neox_args.tokenizer.tokenize(raw_text)

    memorize(neox_args, model, tokens)
