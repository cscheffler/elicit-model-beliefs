import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# Single-word tokens we'll record
WORDS = [
    "Yes",
    "No",
    "True",
    "False",
    "yes",
    "no",
    "true",
    "false",
    "YES",
    "NO",
    "TRUE",
    "FALSE",
]
# The 4 yes/no tokens we'll extract
YES, NO, TRUE, FALSE = 0, 1, 2, 3

# The 8 prompt templates used to ask the same question repeatedly
PROMPT_TEMPLATES = [
    'Consider the claim, "{claim}" Do you believe the claim? Answer in one word — Yes or No.',
    'Consider the claim, "{claim}" Do you believe the claim? Answer in one word — No or Yes.',
    'Consider the claim, "{claim}" Do you believe the claim? Answer in one word — True or False.',
    'Consider the claim, "{claim}" Do you believe the claim? Answer in one word — False or True.',
    "{claim} Yes or No?",
    "{claim} No or Yes?",
    "{claim} True or False?",
    "{claim} False or True?",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment(model_id, expanded_dataset, dtype, batch_size):
    """
    Run the full belief-elicitation pipeline for one model.

    Loads the model, gets the next-token logits for every prompt in
    expanded_dataset, saves the raw logits to `results/elicit-beliefs-<model>.pt`,
    converts them to per-claim affirmation probabilities, and plots a summary.

    Returns (model, tokenizer, logits, p_affirm).
    """
    import time

    start_time = time.time()
    print("Running experiment", model_id, "on", DEVICE, "with dtype", dtype)
    model, tokenizer = load_model(model_id, dtype)
    target_ids = get_true_false_token_ids(tokenizer)
    prompts = expanded_dataset["claim"]
    logits, top_other_logit, top_other_id = first_token_logits(
        prompts, target_ids, model, tokenizer, batch_size
    )
    top_other_token = [
        tokenizer.decode(tid, skip_special_tokens=True) for tid in top_other_id
    ]
    model_slug = model_id.split("/")[-1]
    os.makedirs("results", exist_ok=True)
    torch.save(
        {
            "logits": logits.detach().cpu(),
            "top_other_logit": top_other_logit.detach().cpu(),
            "top_other_id": top_other_id.detach().cpu(),
            "top_other_token": top_other_token,
        },
        f"results/elicit-beliefs-{model_slug}.pt",
    )
    p_affirm = logits_to_affirm_prob(logits, expanded_dataset["label"])
    present_results(p_affirm)
    stop_time = time.time()
    print(f"Experiment took {(stop_time - start_time) / 60:.1f} min")
    return model, tokenizer, logits, p_affirm


def present_results(p_affirm):
    """
    Plot two histograms summarising the affirmation probabilities.

    p_affirm has shape (num_claims, num_templates). For each claim we look at:
      - certainty: how far the mean P(affirm) is from 0.5 (i.e. how decisive
        the model is, in [0.5, 1]).
      - stability: how consistent P(affirm) is across the different prompt
        templates, in [0, 1] (1 = identical answers, lower = more disagreement).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mean_p_affirm = p_affirm.mean(dim=1)
    stdev_p_affirm = p_affirm.std(dim=1)

    certainty = np.vstack((mean_p_affirm, 1 - mean_p_affirm)).max(axis=0)
    stability = 1 - 2 * stdev_p_affirm

    plt.figure()
    plt.title("Density of certainty of yes/no distributions")
    plt.hist(certainty, bins=np.linspace(0.5, 1, 51), density=True, edgecolor="white")
    plt.xlabel("certainty in [0.5, 1]")
    plt.ylabel("density")

    plt.figure()
    plt.title("Belief stability of yes probabilities")
    plt.hist(stability, bins=np.linspace(0, 1, 51), density=True, edgecolor="white")
    plt.xlabel("belief stability in [0, 1]")
    plt.ylabel("density")

    plt.show()


def get_dtype(dtype=torch.float16):
    """
    Use the requested dtype on GPU, but fall back to float32 on CPU.
    """
    return dtype if DEVICE == "cuda" else torch.float32


def load_data():
    """
    Load the dataset.

    The Azaria & Mitchell True-False dataset lives at notrichardren/azaria-mitchell on
    HuggingFace. It has ~13.7k statements across 12 topics (cities, companies, animals,
    elements, facts, inventions, etc.), each labelled 0 (false) or 1 (true).
    """
    dataset = datasets.load_dataset("notrichardren/azaria-mitchell", split="train")
    print(f"Total examples: {len(dataset)}")
    print(
        f"Label distribution: {sum(dataset['label'])} true, {len(dataset) - sum(dataset['label'])} false"
    )
    print("First 5 samples:")
    for i in range(5):
        print(dataset[i])

    # Keep only the two fields we care about, then expand.
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("claim", "label")]
    )

    # Expand the dataset. Each claim becomes 4 prompts, all sharing the original label.
    def expand_batch(batch):
        out_claims, out_labels = [], []
        for claim, label in zip(batch["claim"], batch["label"]):
            for tpl in PROMPT_TEMPLATES:
                out_claims.append(tpl.format(claim=claim))
                out_labels.append(label)
        return {"claim": out_claims, "label": out_labels}

    expanded = dataset.map(
        expand_batch,
        batched=True,
        remove_columns=dataset.column_names,  # replace, don't append
    )
    print(f"Expanded dataset size:  {len(expanded)}")

    return expanded


def load_model(model_id, dtype):
    """
    Load a HuggingFace causal LM and its tokenizer, ready for inference.

    Sets left-padding (so the last token of every prompt lines up at the end
    of the batch) and a pad token if the tokenizer lacks one. Qwen models get
    their context window capped at 2048 to save memory. Returns (model,
    tokenizer) with the model in eval mode on DEVICE.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)
    if "Qwen" in model_id:
        config.max_position_embeddings = 2048

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=dtype,
    ).to(DEVICE)

    model.eval()
    print("Model loaded.")

    return model, tokenizer


def get_true_false_token_ids(tokenizer):
    """
    Target token IDs. Verify these are single tokens.
    """
    target_ids = []
    for w in WORDS:
        ids = tokenizer.encode(w, add_special_tokens=False)
        print(f"{w!r:>8} -> {ids}  decoded: {[tokenizer.decode([t]) for t in ids]}")
        target_ids.append(ids[0])  # first sub-token if it splits
    target_ids = torch.tensor(target_ids)
    return target_ids


def format_prompt(claim_text, tokenizer):
    """
    Wrap a single user message with the chat template, leaving
    the assistant turn open so the next token is the model's reply.
    """
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful but firm assistant. You provide accurate responses to all questions. If the user points out a factual error, correct it if and only if you truly agree with the correction.",
            },
            {"role": "user", "content": claim_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def first_token_logits(prompts, target_ids, model, tokenizer, batch_size):
    """
    Generate distribution over next token for each prompt.

    Returns:
        target_logits: (batch_size, 4) — logits for [Yes, No, True, False]
        top_other_logits: (batch_size,) — largest logit among non-target tokens
        top_other_ids: (batch_size,) — token id of that token
    """
    from tqdm import tqdm

    target_chunks = []
    top_other_logit_chunks = []
    top_other_id_chunks = []

    target_ids_set = target_ids.to(DEVICE)

    for i in tqdm(range(0, len(prompts), batch_size), mininterval=300):
        batch = [format_prompt(p, tokenizer) for p in prompts[i : i + batch_size]]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(DEVICE)
        out = model(**enc)
        last = out.logits[:, -1, :]  # (Batch, Vocab)

        # Target logits
        sliced = last[:, target_ids_set]  # (B, Target_ids)
        target_chunks.append(sliced.float().cpu())

        # Mask out target token positions, then find the argmax
        mask = torch.ones(last.shape[-1], dtype=torch.bool, device=DEVICE)
        mask[target_ids_set] = False
        masked = last[:, mask]  # (B, V - T)

        # Map local argmax back to original vocab indices
        vocab_indices = torch.where(mask)[0]  # (V - T,)
        local_argmax = masked.argmax(dim=-1)  # (B,)

        top_other_logit_chunks.append(
            masked[torch.arange(masked.shape[0]), local_argmax].float().cpu()
        )
        top_other_id_chunks.append(vocab_indices[local_argmax].cpu())

    return (
        torch.cat(target_chunks, dim=0),
        torch.cat(top_other_logit_chunks, dim=0),
        torch.cat(top_other_id_chunks, dim=0),
    )


def logits_to_affirm_logit(logits, labels=None, prompts=None):
    """
    Convert raw target-token logits into log P(model affirms the claim).

    The input logits have one row per expanded prompt (claims interleaved with
    PROMPT_TEMPLATES). For each template we do a 2-way softmax over just the two
    relevant words (Yes/No or True/False) so that "affirm" always means the same
    thing, then reshape to (num_claims, num_templates).

    Args:
        logits: (num_prompts, len(WORDS)) target-token logits from
            first_token_logits.
        labels: optional 0/1 label per expanded row. If given, prints accuracy
            and the mean affirmation probability per label as a sanity check.
        prompts: optional subset/order of template indices to use; defaults to
            all four templates.

    Returns:
        logit_p_affirm: (num_claims, num_templates) logit of the probability the
            model affirms each claim, one column per prompt template.
    """

    if prompts is None:
        prompts = torch.arange(len(PROMPT_TEMPLATES))
    else:
        prompts = torch.tensor(prompts)

    n_claims = logits.shape[0] // len(PROMPT_TEMPLATES)
    indexes = (
        prompts[None, :] + (torch.arange(n_claims) * len(PROMPT_TEMPLATES))[:, None]
    ).flatten()
    logits_by_claim = logits[indexes, :].view(
        n_claims, len(prompts), len(WORDS)
    )  # (Claim, Template, Word)

    def two_way_affirm_logit(logits, pos_col, neg_col):
        """P(positive) from a 2-way softmax over just the two relevant logits."""
        return (
            logits[:, pos_col] - logits[:, neg_col]
        )  # logit(P(positive word)). shape: (C,)
        # pair = torch.stack([logits[:, pos_col], logits[:, neg_col]], dim=1)  # (C, 2)
        # return pair[:, 0] - pair.logsumexp(dim=1)  # log P(positive word)

    # logit P(affirmative) for each of the templates, mapped onto a common axis:
    temp = []
    for i in range(0, logits_by_claim.shape[1], 4):
        temp.append(two_way_affirm_logit(logits_by_claim[:, i], YES, NO))  # Yes or No
        temp.append(
            two_way_affirm_logit(logits_by_claim[:, i + 1], YES, NO)
        )  # Yes or No
        temp.append(
            two_way_affirm_logit(logits_by_claim[:, i + 2], TRUE, FALSE)
        )  # Yes or No
        temp.append(
            two_way_affirm_logit(logits_by_claim[:, i + 3], TRUE, FALSE)
        )  # Yes or No
    logit_p_affirm = torch.stack(temp, dim=1)  # (C, T)

    if labels is not None:
        import pandas as pd

        # Aggregate the prompts per claim:
        p_claim = p_affirm.mean(dim=1)  # mean P(claim is true), averaged over templates

        # Labels (one per claim — they were identical across the expanded data set rows):
        labels = torch.tensor(labels).view(n_claims, len(PROMPT_TEMPLATES))
        assert (labels[:, 0:1] == labels).all(), "labels differ within a claim group"
        labels = labels[:, 0]

        # Quick sanity check: does mean P(affirm) separate true from false claims?
        results = pd.DataFrame(
            {
                "p_yesno_1": p_yesno_1,
                "p_yesno_2": p_yesno_2,
                "p_tf_1": p_tf_1,
                "p_tf_2": p_tf_2,
                "p_affirm_mean": p_claim,
                "p_affirm_std": p_affirm.std(dim=1),  # disagreement across templates
                "label": labels,
            }
        )
        acc = ((p_claim > 0.5).long() == labels).float().mean()
        print(f"Accuracy (threshold 0.5): {acc:.3f}")
        print(results.groupby("label")["p_affirm_mean"].mean())

    return logit_p_affirm


def logits_to_affirm_prob(*args, **kwargs):
    """
    Convert raw target-token logits into P(model affirms the claim). See
    the call signature of `logits_to_affirm_log_prob` for details.
    """
    lgt = logits_to_affirm_logit(*args, **kwargs)
    return torch.special.expit(lgt)


def compute_metrics(p_affirm_logit, logits, top_other_logit):
    """
    Compute all the metrics used to evaluate models for belief stability.

    Args:
        p_affirm_logit: (num_claims, num_templates) log-odds of an affirmative
            response (Yes or True) for each factual claim and prompt template.
        logits: (num_prompts, num_tokens) target-token logits for each prompt.
        top_other_logit: (num_prompts,) logit of the most probable token
            not in WORDS for each prompt.

    Returns:
        logit_p_affirm: (num_claims, num_templates) logit of the probability the
            model affirms each claim, one column per prompt template.
    """
    import numpy as np
    from scipy.special import expit, logsumexp
    from scipy.stats import spearmanr

    p_affirm_logit = np.asarray(p_affirm_logit)  # (Claims, Templates)
    p_affirm = expit(p_affirm_logit)
    mean_p_affirm = p_affirm.mean(axis=1)  # (C,)
    std_p_affirm = p_affirm.std(axis=1)  # (C,)
    certainty = np.vstack((mean_p_affirm, 1 - mean_p_affirm)).max(axis=0)  # (C,)
    logit_stdev = p_affirm_logit.std(axis=1)  # (C,)

    # TODO: Leakage is computed is the sum of all other tokens, excluding
    # Yes, No, True, False, which isn't quite right since we should use either
    # (Yes, No) or (True, False) as viable tokens and not all 4 of them. That
    # depends on which prompt template was used though so, for now, this
    # calculation is good enough.
    all_logits = torch.hstack((logits, top_other_logit[:, None]))  # (C, Words+1)
    all_probs = all_logits.softmax(axis=1)  # (C, W+1)
    leakage = np.array(all_probs[:, 4:].sum(axis=1))  # (C,)

    corr = spearmanr(p_affirm_logit).statistic
    if np.any(np.isnan(corr)):
        corr_eig = np.nan
    else:
        corr_eig = np.linalg.eig(corr)[0][0] / p_affirm_logit.shape[1]

    def logit_mean_expit(x, axis=None):
        x = np.asarray(x, dtype=np.float64)
        sp = np.logaddexp(0.0, x)  # softplus(x), stable
        log_num = logsumexp(x - sp, axis=axis)  # log( sum_i expit(x_i) )
        log_den = logsumexp(-sp, axis=axis)  # log( sum_i (1 - expit(x_i)) )
        return log_num - log_den

    return {
        "certainty_dist": certainty,
        "mean_certainty": np.mean(certainty),
        "logit_mean_dist": logit_mean_expit(p_affirm_logit, axis=1),
        "logit_stdev_dist": logit_stdev,
        "mean_logit_stdev": np.mean(logit_stdev),
        "spearmanr_corr": corr,
        "stability": corr_eig,
        "leakage": np.mean(leakage),
        "stability_v1": np.mean(1 - 2 * std_p_affirm),
        "stability_v1_dist": 1 - 2 * std_p_affirm,
    }


def clear_hf_model_cache(model_id):
    """
    Delete all cached revisions of model_id from the local HuggingFace cache.

    Useful for freeing disk space between models when running several
    experiments in one session.
    """
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == model_id:
            revisions = [rev.commit_hash for rev in repo.revisions]
            delete_strategy = cache_info.delete_revisions(*revisions)
            delete_strategy.execute()
