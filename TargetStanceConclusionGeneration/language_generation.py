from .pplm import *
from .IniReader import *


def load_config(file_path: str, param_key: str):
    reader = IniReader(file_path)
    conf = reader[param_key]
    for k in conf:
        if "." in conf[k]:
            conf[k] = float(conf[k])
        elif conf[k].upper() == "TRUE":
            conf[k] = True
        elif conf[k].upper() == "FALSE":
            conf[k] = False
        else:
            conf[k] = int(conf[k])
    return conf


def full_sentence_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=35,
        stepsize=0,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        own_prints=True,
        **kwargs
):
    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bow_indices(bag_of_words, tokenizer)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []
    org_bow = bag_of_words + []
    org_context = context

    for i in range(num_samples):
        return_word = "This variable is filled with the word, that caused a loop"
        so_far = None
        discrim_loss = None
        loss_in_time = None
        bag_of_words = org_bow + []
        context = org_context + []
        while return_word:
            return_word, so_far, discrim_loss, loss_in_time = generate_text_pplm(
                model=model,
                tokenizer=tokenizer,
                context=context,
                device=device,
                perturb=True,
                bow=bag_of_words,
                bow_vector=bow_indices,
                classifier=classifier,
                class_label=class_id,
                loss_type=loss_type,
                length=length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl_scale,
                verbosity_level=verbosity_level,
                own_prints=own_prints
            )
            context = tokenizer.decode(so_far.tolist()[0]).replace("<|endoftext|>", "")
            context = tokenizer.encode(tokenizer.bos_token + context, add_special_tokens=False)
            if return_word in bag_of_words:
                if own_prints:
                    print(f"Removing word '{return_word}' from BoW!")
                bag_of_words.remove(return_word)
        pert_gen_tok_texts.append(so_far)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return pert_gen_tok_texts


def load_model_and_tokenizer(model_dir: str):
    global MODEL, TOKENIZER, DEVICE
    if MODEL and TOKENIZER and DEVICE:
        return MODEL, TOKENIZER, DEVICE
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    pretrained_model = model_dir
    MODEL = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
    MODEL.to(DEVICE)
    MODEL.eval()
    TOKENIZER = GPT2Tokenizer.from_pretrained(pretrained_model)
    return MODEL, TOKENIZER, DEVICE


def gen(model_dir: str, config_path: str, config_key: str, start: str, words: List[str],
        sentiment: float = 0, own_prints: bool = True) -> str:
    torch.manual_seed(0)
    np.random.seed(0)
    class_label = -1
    discrim = None
    if sentiment >= 0.3:
        class_label = 2
        discrim = "sentiment"
    elif sentiment <= -0.3:
        class_label = 3
        discrim = "sentiment"
    conf = load_config(config_path, config_key)
    model, tokenizer, device = load_model_and_tokenizer(model_dir)
    # ----------------------------------- Taken from run_pplm ----------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    tokenized_cond_text = tokenizer.encode(
        tokenizer.bos_token + start,
        add_special_tokens=False
    )
    # ------------------------------------------------------------------------------------------------------------------
    pert_gen_tok_texts = full_sentence_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        bag_of_words=words,
        discrim=discrim,
        class_label=class_label,
        temperature=1.0,
        verbosity_level=QUIET,
        own_prints=own_prints,
        **conf
    )
    pert_gen_text = ""
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])
    return pert_gen_text.replace("<|endoftext|>", "")
