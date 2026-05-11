# test_capstone.py
#
# This file contains a suite of pytest tests for a capstone project that
# involves building a miniature GPT model. The tests are divided into three
# sections: unit tests for each core component (TextDataset, MiniGPT,
# MiniTrainer) and an integration test to verify that the components work
# together as expected.

import copy

import pytest
import torch

# Attempt to import student's concrete implementations.
# If these imports fail, it means the student has not created the required files
# or has named the classes incorrectly.
try:
    from implementation import MiniGPT, MiniTrainer, TextDataset
except ImportError:
    # Create dummy classes to allow the test file to be parsed by pytest,
    # but all tests will fail. This provides a clearer error message than a
    # raw ImportError.
    class MissingImplementation:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Failed to import one or more required classes: "
                "TextDataset, MiniGPT, MiniTrainer. "
                "Please ensure they are defined in 'implementation.py'."
            )
    TextDataset = MiniGPT = MiniTrainer = MissingImplementation


# --- Fixtures ---

@pytest.fixture
def text_dataset_fixture() -> TextDataset:
    """Provides a consistent, simple TextDataset instance for testing."""
    # A simple, repetitive corpus to make behavior predictable.
    # Vocab: 'a', 'b', 'c', ' '. (size 4)
    raw_text = "abc abc abc abc abc abc"
    # STUDENT_HINT: This fixture creates a TextDataset. If tests using it fail,
    # check your TextDataset constructor and data processing logic.
    return TextDataset(raw_text=raw_text, block_size=4, split_ratio=0.8)


@pytest.fixture
def mini_gpt_fixture(text_dataset_fixture: TextDataset) -> MiniGPT:
    """Provides a consistent, small MiniGPT instance for testing."""
    # Model parameters are kept small for fast test execution.
    config = {
        "vocab_size": text_dataset_fixture.vocab_size,
        "block_size": 4,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 16,
    }
    # STUDENT_HINT: This fixture creates a MiniGPT model. If tests using it fail,
    # check your model's __init__ method and layer definitions.
    return MiniGPT(**config)


@pytest.fixture
def mini_trainer_fixture(mini_gpt_fixture: MiniGPT, text_dataset_fixture: TextDataset) -> MiniTrainer:
    """Provides a basic MiniTrainer instance for testing."""
    # STUDENT_HINT: This fixture wires the model and dataset into the trainer.
    # Failures here could point to issues in the MiniTrainer's constructor.
    return MiniTrainer(model=mini_gpt_fixture, dataset=text_dataset_fixture, learning_rate=1e-3, device='cpu')


# --- Unit Tests for TextDataset ---

def test_textdataset_get_batch_shapes(text_dataset_fixture: TextDataset):
    """
    Verifies that get_batch() returns tensors of the correct shape.
    Why: The model and training loop rely on consistently shaped input and target tensors.
    """
    batch_size = 16
    xb, yb = text_dataset_fixture.get_batch('train', batch_size=batch_size)

    expected_shape = (batch_size, text_dataset_fixture.block_size)
    assert xb.shape == expected_shape, f"Input batch 'xb' has wrong shape. Expected {expected_shape}, got {xb.shape}"
    assert yb.shape == expected_shape, f"Target batch 'yb' has wrong shape. Expected {expected_shape}, got {yb.shape}"
    # STUDENT_HINT: This test checks the shape of the tensors from `get_batch`.
    # If it fails, check the logic where you stack the individual sequences into a batch tensor.


def test_textdataset_token_values_in_range(text_dataset_fixture: TextDataset):
    """
    Verifies that all token IDs in a batch are valid indices for the vocabulary.
    Why: Token IDs outside the vocabulary range will cause embedding lookups to fail.
    """
    xb, yb = text_dataset_fixture.get_batch('train', batch_size=16)
    vocab_size = text_dataset_fixture.vocab_size

    assert (xb >= 0).all() and (xb < vocab_size).all(), f"Input batch 'xb' contains token IDs out of vocabulary range [0, {vocab_size-1}]"
    assert (yb >= 0).all() and (yb < vocab_size).all(), f"Target batch 'yb' contains token IDs out of vocabulary range [0, {vocab_size-1}]"
    # STUDENT_HINT: This test ensures token IDs are valid. If it fails, review your
    # character-to-integer mapping (encoder) and the tokenization process.


def test_textdataset_batch_content_is_shifted(text_dataset_fixture: TextDataset):
    """
    Verifies that the target tensor 'yb' is the input tensor 'xb' shifted by one position.
    Why: This relationship is fundamental to next-token prediction, where the model
    learns to predict the next token in the sequence.
    """
    xb, yb = text_dataset_fixture.get_batch('train', batch_size=8)
    # For any sequence `i` in the batch, `yb[i, j]` should be the token that
    # follows `xb[i, j]` in the original text. This means `yb` should be a
    # shifted version of `xb`.
    are_equal = torch.equal(xb[:, 1:], yb[:, :-1])
    assert are_equal, "Target batch 'yb' is not a one-step-shifted version of 'xb'. yb[:, :-1] should equal xb[:, 1:]"
    # STUDENT_HINT: This test checks the core logic of next-token prediction setup.
    # `yb` should be the sequence that comes immediately after `xb`. If this fails,
    # re-examine how you construct the `x` and `y` tensors from the raw data in `get_batch`.


def test_textdataset_encoder_decoder_roundtrip(text_dataset_fixture: TextDataset):
    """
    Verifies that encoding a string and then decoding it results in the original string.
    Why: A consistent encoding/decoding scheme is essential for interpreting the
    model's input and output.
    """
    original_text = "abc"
    encoder = text_dataset_fixture.encoder
    decoder = text_dataset_fixture.decoder

    encoded = encoder(original_text)
    decoded_text = decoder(encoded)

    assert isinstance(encoded, list) and all(isinstance(i, int) for i in encoded), "Encoder should return a list of integers."
    assert isinstance(decoded_text, str), "Decoder should return a string."
    assert original_text == decoded_text, f"Encoder/Decoder roundtrip failed. Expected '{original_text}', got '{decoded_text}'"
    # STUDENT_HINT: This test checks your `encoder` and `decoder` functions.
    # Make sure your string-to-int and int-to-string mappings are perfectly symmetric.


# --- Unit Tests for MiniGPT ---

def test_minigpt_forward_pass_shapes(mini_gpt_fixture: MiniGPT):
    """
    Verifies the `forward` method returns logits of the correct shape and no loss when targets are None.
    Why: The output logits tensor must have dimensions corresponding to (batch, time, vocab_size)
    for the loss function and generation logic to work correctly.
    """
    batch_size = 8
    block_size = mini_gpt_fixture.block_size
    vocab_size = mini_gpt_fixture.vocab_size

    # Create a dummy input tensor of token indices
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    logits, loss = mini_gpt_fixture(idx, targets=None)

    expected_logits_shape = (batch_size, block_size, vocab_size)
    assert logits.shape == expected_logits_shape, f"Logits shape is incorrect. Expected {expected_logits_shape}, got {logits.shape}"
    assert loss is None, f"Loss should be None when targets are not provided, but it was not."
    # STUDENT_HINT: This test checks the output shape of your model's `forward` pass.
    # The final layer in your model should be a Linear layer that maps the embedding dimension to the vocabulary size.


def test_minigpt_forward_pass_with_loss(mini_gpt_fixture: MiniGPT):
    """
    Verifies the `forward` method returns a scalar loss value when targets are provided.
    Why: The calculated loss is the primary signal for training. It must be a single,
    scalar value for backpropagation.
    """
    batch_size = 8
    block_size = mini_gpt_fixture.block_size
    vocab_size = mini_gpt_fixture.vocab_size

    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))

    _, loss = mini_gpt_fixture(idx, targets=targets)

    assert loss is not None, "Loss should not be None when targets are provided."
    assert isinstance(loss, torch.Tensor), f"Loss should be a torch.Tensor, but got {type(loss)}"
    assert loss.ndim == 0, f"Loss must be a scalar (0-dimensional tensor), but its shape is {loss.shape}"
    # STUDENT_HINT: This test checks if loss calculation is correct. You should use `torch.nn.functional.cross_entropy`.
    # Remember to reshape your logits and targets to (B*T, C) and (B*T) respectively before passing them to the loss function.


def test_minigpt_causal_attention_mask(mini_gpt_fixture: MiniGPT):
    """
    Verifies that the model's output for a given token is not influenced by future tokens.
    Why: This is the "auto-regressive" property of a GPT. A token's prediction can only
    depend on itself and the tokens that came before it.
    """
    mini_gpt_fixture.eval()  # Set to evaluation mode to disable dropout for determinism
    batch_size = 1
    block_size = mini_gpt_fixture.block_size
    vocab_size = mini_gpt_fixture.vocab_size

    # Create two input sequences that are identical except for the very last token.
    base_seq = torch.randint(0, vocab_size, (batch_size, block_size))
    seq1 = base_seq.clone()
    seq2 = base_seq.clone()
    seq2[0, -1] = (seq2[0, -1] + 1) % vocab_size  # Change the last token

    # The output for the second-to-last token should be identical for both sequences.
    # A change in the *future* (last token) should not affect the prediction for the present (second-to-last token).
    logits1, _ = mini_gpt_fixture(seq1)
    logits2, _ = mini_gpt_fixture(seq2)

    # We check the logits at the second to last position (index -2)
    logits_at_penultimate_step1 = logits1[0, -2, :]
    logits_at_penultimate_step2 = logits2[0, -2, :]

    assert torch.allclose(logits_at_penultimate_step1, logits_at_penultimate_step2, atol=1e-6), \
        "Changing a future token affected a past token's logits. Check your causal attention mask."
    # STUDENT_HINT: This is a critical test for the Transformer block. If it fails, your
    # causal self-attention mask is likely incorrect. Ensure you are using `torch.tril`
    # to create a lower-triangular mask and applying it correctly before the softmax in the attention mechanism.


# --- Unit Tests for MiniTrainer ---

def test_minitrainer_train_updates_model_weights(mini_trainer_fixture: MiniTrainer):
    """
    Verifies that calling train() for a few steps actually changes the model's parameters.
    Why: If the weights don't change, the model is not learning. This tests the basic
    mechanics of the training loop (forward, backward, step).
    """
    model = mini_trainer_fixture.model
    # Deepcopy the initial state of the model's parameters
    initial_params = [p.clone().detach() for p in model.parameters()]

    # Run training for a few iterations
    mini_trainer_fixture.train(max_iters=3, batch_size=2)

    # Get the new parameters
    final_params = [p.clone().detach() for p in model.parameters()]

    # Check that at least one parameter tensor has changed
    params_have_changed = any(not torch.equal(p_init, p_final) for p_init, p_final in zip(initial_params, final_params))

    assert params_have_changed, "Model parameters did not change after a few training steps. Check optimizer.step() and loss.backward()."
    # STUDENT_HINT: This test ensures your training loop works. If it fails, check these things in your `train` method:
    # 1. Are you calling `optimizer.zero_grad()`?
    # 2. Are you calling `loss.backward()`?
    # 3. Are you calling `optimizer.step()`?


def test_minitrainer_train_completes_without_error(mini_trainer_fixture: MiniTrainer):
    """
    Verifies that the train method can run for a few iterations, including evaluation, without crashing.
    Why: This is a basic smoke test to catch runtime errors in the training or evaluation loops.
    """
    try:
        mini_trainer_fixture.train(max_iters=2, eval_interval=1, batch_size=2)
    except Exception as e:
        pytest.fail(f"trainer.train() raised an exception: {e}")
    # STUDENT_HINT: This is a "smoke test." If it fails, there's a runtime error in your `train` loop.
    # Look at the full error traceback to identify the line causing the problem. It could be in data fetching,
    # the model forward pass, or the evaluation logic.


def test_minitrainer_zero_lr_does_not_update_weights(mini_gpt_fixture: MiniGPT, text_dataset_fixture: TextDataset):
    """
    Verifies that with a learning rate of 0, the model's weights do not change.
    Why: This confirms that the optimizer and learning rate are correctly wired and that
    weight updates are a direct result of the optimization step, not some other side effect.
    """
    # Create a trainer with a learning rate of 0.0
    trainer = MiniTrainer(model=mini_gpt_fixture, dataset=text_dataset_fixture, learning_rate=0.0)
    model = trainer.model

    initial_params = [p.clone().detach() for p in model.parameters()]
    trainer.train(max_iters=3, batch_size=2)
    final_params = [p.clone().detach() for p in model.parameters()]

    # Check that NO parameter tensors have changed
    params_are_identical = all(torch.equal(p_init, p_final) for p_init, p_final in zip(initial_params, final_params))

    assert params_are_identical, "Model parameters changed even with a learning rate of 0.0."
    # STUDENT_HINT: This is a sanity check. If parameters change with LR=0, something is fundamentally
    # wrong. It's unlikely to fail if the previous trainer test passes, but it's good practice.


# --- Integration Test ---

@pytest.mark.integration
def test_integration_training_loop_improves_model():
    """
    Verifies the end-to-end process: training the model on the dataset reduces the loss.
    Why: This is the most important test. It confirms that all components (dataset, model,
    trainer) work together to achieve the fundamental goal of machine learning: improving
    a model's performance on a task.
    """
    # 1. Setup: Create a small, deterministic training environment
    torch.manual_seed(1337)
    corpus = "hello world. this is a simple test. hello world again."
    block_size = 8
    
    # STUDENT_HINT: This integration test uses all your components together.
    # A failure here means the pieces don't fit, even if they pass unit tests.
    # Read the failure message carefully to see which assertion failed.

    try:
        dataset = TextDataset(corpus, block_size=block_size, split_ratio=0.8)
        model = MiniGPT(vocab_size=dataset.vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=32)
        trainer = MiniTrainer(model, dataset, learning_rate=1e-2, device='cpu')
    except Exception as e:
        pytest.fail(f"Failed during setup of integration test: {e}")

    # Capture initial model state
    initial_params = [p.clone().detach() for p in model.parameters()]

    # 2. Evaluate initial performance on a fixed validation batch
    # We must use the same batch for initial and final loss to ensure a fair comparison.
    val_xb, val_yb = dataset.get_batch('val', batch_size=8)
    model.eval()
    _, initial_loss = model(val_xb, val_yb)
    model.train()
    assert initial_loss is not None, "Initial loss calculation failed."

    # 3. Train the model
    trainer.train(max_iters=100, eval_interval=50, batch_size=8)

    # 4. Evaluate final performance on the SAME validation batch
    model.eval()
    _, final_loss = model(val_xb, val_yb)
    model.train()
    assert final_loss is not None, "Final loss calculation failed."

    # Capture final model state
    final_params = [p.clone().detach() for p in model.parameters()]

    # 5. Assertions
    # Assertion 1: Model parameters must have changed.
    params_have_changed = any(not torch.equal(p_init, p_final) for p_init, p_final in zip(initial_params, final_params))
    assert params_have_changed, "Integration Test Failed: Model parameters did not change after training."
    # STUDENT_HINT: If this fails, your trainer's `optimizer.step()` might not be working correctly.

    # Assertion 2: The validation loss must decrease by a meaningful amount (e.g., 10%).
    # This shows the model is actually learning, not just changing weights randomly.
    assert final_loss < initial_loss * 0.9, \
        f"Integration Test Failed: Loss did not decrease significantly. Initial: {initial_loss.item():.4f}, Final: {final_loss.item():.4f}"
    # STUDENT_HINT: If loss doesn't decrease, the problem is deep. It could be:
    # - The model architecture is flawed (e.g., bad activation functions, no non-linearity).
    # - The data loading is incorrect (e.g., `xb` and `yb` don't align).
    # - The loss calculation in the model's forward pass is wrong.
    # - The learning rate is too high or too low.

    # Assertion 3: Generate some text to see if it's learned basic patterns.
    # This is a "soft" check for basic coherence.
    start_context = torch.zeros((1, 1), dtype=torch.long, device='cpu') # Start with the first token in vocab
    
    # Simple generation loop
    generated_indices = model.generate(start_context, max_new_tokens=20)[0].tolist()
    generated_text = dataset.decoder(generated_indices)
    
    # Check for non-randomness: a simple check is to see if any 2-character
    # substring of the generated text also exists in the original corpus.
    corpus_bigrams = {corpus[i:i+2] for i in range(len(corpus) - 1)}
    generated_bigrams = {generated_text[i:i+2] for i in range(len(generated_text) - 1)}
    
    assert len(generated_text) > 1, "Generation produced an empty or single-character string."
    assert len(corpus_bigrams.intersection(generated_bigrams)) > 0, \
        f"Integration Test Failed: Generated text '{generated_text}' shows no patterns from the training corpus '{corpus}'"
    # STUDENT_HINT: If this fails, it means the model isn't learning patterns. This is related to the loss
    # not decreasing. Check that your generation logic correctly samples from the model's output logits.