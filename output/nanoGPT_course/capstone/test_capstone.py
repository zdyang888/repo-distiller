# test_capstone.py
#
# Pytest test file for the MiniGPT capstone project.
# This file contains unit tests for each individual component (Config, DataLoader,
# Model, Trainer) and a final integration test to verify the end-to-end
# functionality of the system.
#
# To run: `pytest -v`
# To run integration test only: `pytest -v -m integration`

import pytest
import torch
import torch.nn as nn
from pathlib import Path

# Attempt to import student's code from the 'implementation' directory/module
try:
    from implementation import (
        MiniGPTConfig,
        IMiniGPTModel,
        IMiniTextDataLoader,
        IMiniTrainer,
    )
except ImportError:
    # Create dummy classes if the implementation is not found,
    # allowing tests to be collected but fail with a clear message.
    class MiniGPTConfig: pass
    class IMiniGPTModel(nn.Module): pass # Inherit to avoid errors in fixture
    class IMiniTextDataLoader: pass
    class IMiniTrainer: pass
    pytest.fail(
        "Could not import classes from 'implementation'. "
        "Ensure your implementation file exists and contains the required classes."
    )


# --- Helper Fixtures ---------------------------------------------------------

@pytest.fixture(autouse=True)
def set_seed():
    """
    Fixture to set a random seed for torch, ensuring test reproducibility.
    This is applied automatically to every test.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture(scope="module")
def dummy_data_file(tmp_path_factory) -> Path:
    """
    Creates a temporary dummy text file for the data loader tests.
    This fixture has a 'module' scope, so the file is created once per test session.
    """
    # A simple, repetitive text to ensure vocabulary is small and patterns are learnable.
    # The text must be long enough for the 10% validation split to be larger than
    # block_size + 1, as required by the reference implementation's get_batch method
    # to avoid raising an error. With block_size=8, we need len(val) >= 9, so total
    # length must be >= 90.
    content = "hello world. this is a test. hello world again. test this is. " * 5
    file_path = tmp_path_factory.mktemp("data") / "input.txt"
    file_path.write_text(content)
    return file_path


# --- Component Fixtures ------------------------------------------------------

@pytest.fixture
def config() -> MiniGPTConfig:
    """
    Provides a minimal, valid MiniGPTConfig instance for testing.
    Using small dimensions to speed up test execution.
    """
    return MiniGPTConfig(
        block_size=16,
        vocab_size=50,  # A reasonable default, large enough for char-level tokenization
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0
    )

@pytest.fixture
def data_loader(dummy_data_file: Path) -> IMiniTextDataLoader:
    """
    Provides a MiniTextDataLoader instance initialized with the dummy data file.
    """
    # Ensure the class can be instantiated before proceeding
    if not issubclass(IMiniTextDataLoader, object) or IMiniTextDataLoader is object:
         pytest.skip("Skipping DataLoader tests: concrete implementation not found.")
    return IMiniTextDataLoader(data_path=str(dummy_data_file), block_size=8, batch_size=4)

@pytest.fixture
def model(config: MiniGPTConfig) -> IMiniGPTModel:
    """
    Provides a MiniGPTModel instance based on the minimal test config.
    """
    if not issubclass(IMiniGPTModel, nn.Module) or IMiniGPTModel is nn.Module:
        pytest.skip("Skipping Model tests: concrete implementation not found.")
    # The config fixture provides all necessary parameters. Its vocab_size is
    # intentionally large enough to be compatible with the data_loader fixture
    # for use in the trainer tests.
    return IMiniGPTModel(config)

@pytest.fixture
def trainer(model: IMiniGPTModel, data_loader: IMiniTextDataLoader) -> IMiniTrainer:
    """
    Provides a MiniTrainer instance with a model, data loader, and default training config.
    """
    if not issubclass(IMiniTrainer, object) or IMiniTrainer is object:
         pytest.skip("Skipping Trainer tests: concrete implementation not found.")
    training_config = {
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'device': 'cpu' # Force CPU for testing to ensure consistency
    }
    return IMiniTrainer(model, data_loader, training_config)


# --- Unit Tests --------------------------------------------------------------

class TestMiniGPTConfig:
    """Unit tests for the MiniGPTConfig dataclass."""

    def test_instantiation_defaults(self):
        """
        Verifies that the dataclass can be instantiated with default values.
        This confirms the dataclass is defined correctly.
        """
        # STUDENT_HINT: This test will fail if the MiniGPTConfig class is not
        # defined as a dataclass or if the default values are incorrect.
        try:
            config = MiniGPTConfig()
            assert config.block_size == 256, f"Expected default block_size 256, got {config.block_size}"
            assert config.vocab_size == 50257, f"Expected default vocab_size 50257, got {config.vocab_size}"
            assert config.n_layer == 6, f"Expected default n_layer 6, got {config.n_layer}"
        except Exception as e:
            pytest.fail(f"Failed to instantiate MiniGPTConfig with default values: {e}")

    def test_instantiation_custom_values(self):
        """
        Verifies that custom values correctly override the defaults during instantiation.
        This checks the basic functionality of a dataclass.
        """
        # STUDENT_HINT: If this fails, check that the attributes of your
        # MiniGPTConfig dataclass are correctly named and typed.
        custom_config = MiniGPTConfig(
            block_size=128,
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.5
        )
        assert custom_config.block_size == 128, "Custom block_size was not set correctly."
        assert custom_config.vocab_size == 100, "Custom vocab_size was not set correctly."
        assert custom_config.dropout == 0.5, "Custom dropout was not set correctly."


class TestMiniTextDataLoader:
    """Unit tests for the MiniTextDataLoader component."""

    def test_get_batch_shape(self, data_loader: IMiniTextDataLoader):
        """
        Verifies that get_batch() returns two tensors of the correct shape [batch_size, block_size].
        This is a fundamental contract of the data loader.
        """
        # STUDENT_HINT: Check your get_batch method. It should return two tensors,
        # 'x' (inputs) and 'y' (targets), both with shape (batch_size, block_size).
        batch_size = 4
        block_size = 8
        x, y = data_loader.get_batch('train')

        expected_shape = (batch_size, block_size)
        assert x.shape == expected_shape, f"Expected input batch shape {expected_shape}, but got {x.shape}"
        assert y.shape == expected_shape, f"Expected target batch shape {expected_shape}, but got {y.shape}"

    def test_get_vocab_size(self, data_loader: IMiniTextDataLoader):
        """
        Verifies that get_vocab_size() returns the correct number of unique tokens.
        This test confirms that the vocabulary has been created correctly from the input text.
        """
        # STUDENT_HINT: Your data loader's __init__ method should read the text,
        # find all unique characters, and store the count. `get_vocab_size` should return this count.
        # The dummy text is "hello world. this is a test. hello world again. test this is."
        # Unique chars: h, e, l, o,  , w, r, d, ., t, i, s, a, g, n -> 15 unique chars
        expected_vocab_size = 15
        actual_vocab_size = data_loader.get_vocab_size()
        assert actual_vocab_size == expected_vocab_size, \
            f"Expected vocab size of {expected_vocab_size} for the dummy text, but got {actual_vocab_size}"

    def test_batch_content_is_valid(self, data_loader: IMiniTextDataLoader):
        """
        Verifies that all token IDs in a batch are within the valid range [0, vocab_size-1].
        This ensures the tokenization process is working as expected.
        """
        # STUDENT_HINT: After tokenizing the text, all token IDs in your data tensors
        # should be less than the vocabulary size. Check your encoding/tokenization logic.
        vocab_size = data_loader.get_vocab_size()
        x, y = data_loader.get_batch('train')
        assert torch.all(x >= 0) and torch.all(x < vocab_size), "Input batch contains out-of-vocabulary token IDs."
        assert torch.all(y >= 0) and torch.all(y < vocab_size), "Target batch contains out-of-vocabulary token IDs."

    def test_target_is_shifted_input(self, tmp_path):
        """
        Verifies that the target tensor 'y' is a one-step-ahead version of the input 'x'.
        This is critical for training a language model to predict the next token.
        """
        # STUDENT_HINT: In `get_batch`, for a given sequence chunk `x`, the target `y`
        # should be the same chunk shifted one position to the left. For example, if
        # x is tokens [0, 1, 2, 3], y should be [1, 2, 3, 4].
        # The reference implementation randomly samples a starting position. To make this
        # test deterministic, we create text just long enough that there is only one
        # possible starting index for the 'train' split.
        block_size = 5
        # For block_size=5, we need len(train_data) >= 6. With a 90/10 split,
        # len(content)=7 gives len(train_data)=6. This forces the start index to be 0.
        deterministic_content = "0123456"
        file_path = tmp_path / "deterministic.txt"
        file_path.write_text(deterministic_content)

        loader = IMiniTextDataLoader(data_path=str(file_path), block_size=block_size, batch_size=1)
        # Create a mapping from char to int to verify tokenization
        chars = sorted(list(set(deterministic_content)))
        stoi = { ch:i for i,ch in enumerate(chars) }

        x, y = loader.get_batch('train')

        # The batch must be tokens for "01234" and "12345"
        expected_x = torch.tensor([[stoi['0'], stoi['1'], stoi['2'], stoi['3'], stoi['4']]], dtype=torch.long)
        expected_y = torch.tensor([[stoi['1'], stoi['2'], stoi['3'], stoi['4'], stoi['5']]], dtype=torch.long)

        assert torch.equal(x, expected_x), f"Expected input x to be tokens for '01234', but got different values."
        assert torch.equal(y, expected_y), f"Expected target y to be tokens for '12345', but got different values."


class TestMiniGPTModel:
    """Unit tests for the MiniGPTModel component."""

    def test_forward_pass_logits_shape(self, model: IMiniGPTModel, config: MiniGPTConfig):
        """
        Verifies the forward pass returns logits with the correct shape.
        The shape should be [batch_size, block_size, vocab_size].
        """
        # STUDENT_HINT: Your model's `forward` method must return logits. The final layer
        # of your model should be a linear layer that projects to the vocabulary size.
        batch_size = 4
        # Create a dummy input tensor of token indices
        idx = torch.randint(0, config.vocab_size, (batch_size, config.block_size))
        logits, loss = model(idx)

        expected_shape = (batch_size, config.block_size, config.vocab_size)
        assert logits.shape == expected_shape, f"Expected logits shape {expected_shape}, but got {logits.shape}"
        assert loss is None, "Loss should be None when targets are not provided."

    def test_forward_pass_loss_calculation(self, model: IMiniGPTModel, config: MiniGPTConfig):
        """
        Verifies that when targets are provided, the forward pass returns a valid scalar loss.
        This checks the loss computation logic.
        """
        # STUDENT_HINT: In the `forward` method, if `targets` are not None, you must
        # calculate the cross-entropy loss between your logits and the targets.
        batch_size = 4
        idx = torch.randint(0, config.vocab_size, (batch_size, config.block_size))
        targets = torch.randint(0, config.vocab_size, (batch_size, config.block_size))

        logits, loss = model(idx, targets)
        assert loss is not None, "Loss should not be None when targets are provided."
        assert isinstance(loss, torch.Tensor), f"Loss must be a torch.Tensor, but got {type(loss)}"
        assert loss.dim() == 0, f"Loss must be a scalar (0 dimensions), but got {loss.dim()} dimensions."

    def test_generate_output_shape(self, model: IMiniGPTModel, config: MiniGPTConfig):
        """
        Verifies that the generate method produces an output tensor of the correct length.
        The new length should be the original length plus max_new_tokens.
        """
        # STUDENT_HINT: The `generate` method should loop `max_new_tokens` times,
        # appending one new token to the input sequence in each iteration.
        initial_context = torch.zeros((1, 5), dtype=torch.long) # A prompt of 5 tokens
        max_new_tokens = 10
        generated_tokens = model.generate(initial_context, max_new_tokens)

        expected_length = initial_context.shape[1] + max_new_tokens
        assert generated_tokens.shape[1] == expected_length, \
            f"Expected generated sequence length {expected_length}, but got {generated_tokens.shape[1]}"

    def test_get_num_parameters(self, model: IMiniGPTModel):
        """
        Verifies that get_num_parameters returns a positive integer count.
        This is a sanity check that the model has trainable parameters.
        """
        # STUDENT_HINT: This method should iterate through `self.parameters()` and sum
        # up the number of elements for each parameter where `requires_grad` is True.
        num_params = model.get_num_parameters()
        assert isinstance(num_params, int), "Number of parameters must be an integer."
        assert num_params > 0, "Model should have at least one trainable parameter."


class TestMiniTrainer:
    """Unit tests for the MiniTrainer component."""

    def test_evaluate_returns_float_and_restores_mode(self, trainer: IMiniTrainer):
        """
        Verifies that evaluate() returns a float and that the model is returned to train mode.
        This checks the basic contract of the evaluation method.
        """
        # STUDENT_HINT: Your `evaluate` method should use a `torch.no_grad()` context,
        # set the model to eval mode (`model.eval()`), and set it back to train mode
        # (`model.train()`) before returning the average loss.
        trainer.model