import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

import discord
from discord.ext import commands
from discord.commands import SlashCommandGroup

from resources.shared import CONTEXTS, INTEGRATION_TYPES

from scripts.tools import journal
from scripts.tools.utility import isDeveloper

from .config import DATA_FILE, MODEL_NAME, MODEL_PATH, LOG_COMPONENT


class MythosTranslatorView(discord.ui.DesignerView):
	def __init__(self, translation=""):
		super().__init__(timeout=None)

		container = discord.ui.Container(colour=discord.Colour.blurple())

		title_text = discord.ui.TextDisplay("### Mythos ML")
		container.add_item(title_text)

		translation_text_display = discord.ui.TextDisplay(translation)
		container.add_item(translation_text_display)

		super().add_item(container)


class MythosMLTranslater(commands.Cog):
	model = None
	tokenizer = None
	device = "cpu"

	command_group = SlashCommandGroup("mythosml", "The MythosML translator", contexts=CONTEXTS, integration_types=INTEGRATION_TYPES)

	def __init__(self, bot):
		self.bot = bot

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		journal.log(f"Using device: {self.device}", 6, component=LOG_COMPONENT)

		self.load_model()

	@commands.message_command(name="De-Mythos", contexts=CONTEXTS, integration_types=INTEGRATION_TYPES)
	async def de_mythos(self, ctx: discord.ApplicationCommand, message: discord.Message):
		authorName = message.author.display_name
		author     = message.author.id
		text       = message.content
		avat       = message.author.display_avatar.url

		await ctx.respond(view=MythosTranslatorView(self.translate(text)))

	@command_group.command(name="train", description="Trains the MythosML machine learning model.")
	@isDeveloper()
	async def train(self, ctx: discord.ApplicationContext):
		await ctx.defer()

		await ctx.respond("Training model...")

		journal.log(f"User {ctx.user} triggered model training", 5, component=LOG_COMPONENT)

		await self.train_model(ctx)

		await ctx.edit(content="Finished training!")

	def translate(self, text: str) -> str:
		"""Runs the translator model on input text"""
		inputs = self.tokenizer(
			text,
			return_tensors="pt",
			max_length=256,
			truncation=True,
			padding="max_length"
		)

		# Move input tensors to the same device as model
		inputs = {k: v.to(self.device) for k, v in inputs.items()}

		with torch.no_grad():
			output_ids = self.model.generate(
				**inputs,
				max_length=256,
				num_beams=4,
				early_stopping=True
			)

		return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

	def load_model(self) -> bool:
		# ===== LOAD MODEL + TOKENIZER =====
		journal.log("Loading model...", 6, component=LOG_COMPONENT)
		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
		self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
		self.model = self.model.to(self.device)
		self.model.eval()
		journal.log("Model loaded", 6, component=LOG_COMPONENT)

	async def train_model(self, ctx: discord.ApplicationContext):
		# Device and precision settings
		# If GPU is available, use mixed precision (bf16) and TF32 for performance
		# Otherwise, fallback to CPU with float32
		if torch.cuda.is_available():
			dtype = torch.bfloat16           # Mixed precision reduces memory usage and speeds up training
			use_bf16, use_tf32 = True, False  # Enable bf16 and TensorFloat32
			torch.backends.cudnn.benchmark = True  # Auto-tune GPU kernels for better throughput
			journal.log("[Training] GPU detected: CUDA + bf16", 5, component=LOG_COMPONENT)
		else:
			dtype = torch.float32            # CPU only supports float32
			use_bf16, use_tf32 = False, False
			journal.log("[Training] No GPU detected: CPU + float32", 5, component=LOG_COMPONENT)

		# Training hyperparameters
		# These can be tuned based on dataset size and GPU memory
		TRAINING_CONFIG = {
			"per_device_train_batch_size": 16,  # Batch size per GPU/CPU core
			"gradient_accumulation_steps": 4,   # Accumulate gradients over multiple steps

			"num_train_epochs": 22,             # Total training epochs
			"learning_rate": 4e-4,              # AdamW learning rate
			"warmup_steps": 8,                  # Number of steps to gradually increase LR

			"optim": "adamw_torch_fused",       # Fused AdamW optimizer for speed on GPU
			"lr_scheduler_type": "linear",      # Linear LR decay

			"bf16": use_bf16,                   # Use bf16 if supported
			"tf32": use_tf32,                   # Use TF32 for faster matmul
			"fp16": False,                      # Disable fp16 (we're using bf16 instead)

			"save_strategy": "no",              # Disable checkpoint saving during training
			"save_steps": 0,
			"save_total_limit": 0,
			"eval_strategy": "no",              # Disable evaluation during training
			"logging_steps": 5,                 # Log every N steps

			"dataloader_num_workers": 0,        # Number of subprocesses for data loading
			"dataloader_pin_memory": False,     # Pin memory to GPU (can improve speed)
			"remove_unused_columns": False      # Keep all dataset columns (necessary for collator)
		}

		# ===== LOAD MODEL & TOKENIZER =====
		journal.log(f"[Training] Loading model: {MODEL_NAME}...", 5, component=LOG_COMPONENT)
		await ctx.edit(content="Loading model...")
		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
		self.model.config.use_cache = False  # Required when using gradient checkpointing
		self.model.to(self.device)
		journal.log("[Training] Model loading complete.", 5, component=LOG_COMPONENT)

		# ===== LOAD DATASET =====
		journal.log("[Training] Loading dataset...", 5, component=LOG_COMPONENT)
		await ctx.edit(content="Loading dataset...")
		dataset = load_dataset("json", data_files=DATA_FILE)["train"]
		journal.log(f"[Training] Dataset loaded: {len(dataset)} samples", 5, component=LOG_COMPONENT)

		# ===== TOKENIZATION =====
		def preprocess(example):
			# Convert raw text to token IDs
			model_input = self.tokenizer(example["input"], max_length=256, truncation=True)
			labels = self.tokenizer(example["output"], max_length=256, truncation=True)["input_ids"]
			model_input["labels"] = torch.tensor(labels, dtype=torch.long)  # Labels must be tensor
			return model_input

		journal.log("[Training] Tokenizing dataset...", 5, component=LOG_COMPONENT)
		await ctx.edit(content="Tokenizing dataset...")
		tokenized_dataset = dataset.map(preprocess, batched=False)
		tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
		tokenized_dataset.set_format(type="torch")
		journal.log("[Training] Tokenization complete.", 5, component=LOG_COMPONENT)

		# ===== DATA COLLATOR =====
		class FastDataCollator:
			"""Pads input_ids, attention_mask, and labels dynamically per batch"""
			def __init__(self, tokenizer):
				self.pad_token_id = tokenizer.pad_token_id
				self.label_pad_token_id = -100  # HF ignores -100 when computing loss

			def __call__(self, features):
				input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True,
										 padding_value=self.pad_token_id)
				attention_mask = pad_sequence([f.get("attention_mask", torch.ones_like(f["input_ids"]))
											  for f in features], batch_first=True, padding_value=0)
				labels = pad_sequence([f["labels"] for f in features], batch_first=True,
									  padding_value=self.label_pad_token_id)
				return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

		data_collator = FastDataCollator(self.tokenizer)

		# ===== TRAINING ARGUMENTS =====
		training_args = TrainingArguments(
			output_dir=MODEL_PATH,
			**TRAINING_CONFIG
		)

		# ===== TRAINER =====
		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=tokenized_dataset,
			tokenizer=self.tokenizer,
			data_collator=data_collator,
		)

		# ===== TRAIN =====
		journal.log("[Training] Starting training...", 5, component=LOG_COMPONENT)
		await ctx.edit(content="Training model...")
		trainer.train()
		journal.log("[Training] Training complete.", 5, component=LOG_COMPONENT)

		# ===== SAVE MODEL =====
		journal.log("[Training] Saving final model...", 5, component=LOG_COMPONENT)
		await ctx.edit(content="Saving model...")
		trainer.save_model(MODEL_PATH)
		self.tokenizer.save_pretrained(MODEL_PATH)
		journal.log(f"[Training] Model saved to {MODEL_PATH}", 5, component=LOG_COMPONENT)


def setup(bot):
	bot.add_cog(MythosMLTranslater(bot))


def teardown(bot):
	bot.remove_cog("MythosMLTranslater")
