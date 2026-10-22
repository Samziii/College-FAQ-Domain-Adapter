# College FAQ Domain Adapter using PEFT

## Project overview
Fine-tunes `google/flan-t5-small` with LoRA (PEFT) to generate college FAQ answers. This is a mini-project demonstrating parameter-efficient fine-tuning, not a production-ready student support system.

## Technology stack
- Python and Google Colab
- PyTorch, Hugging Face Transformers and Datasets
- PEFT (LoRA), Pandas, scikit-learn

## Dataset
- **Rows:** {len(df)}
- **Underlying FAQ intents:** {df['intent_id'].nunique()}
- **Categories:** Admissions, Examinations, Fees, Academic Procedures, Internships, Library and Facilities
- **Split:** train {len(train_df)} rows / {train_df['intent_id'].nunique()} intents; validation {len(val_df)} / {val_df['intent_id'].nunique()}; test {len(test_df)} / {test_df['intent_id'].nunique()}

Each intent has the original question plus prefixed paraphrases. These are not independent facts. Splitting by intent prevents variants of the same FAQ from leaking between splits.

## Model and training
- Base model: `google/flan-t5-small`
- Method: LoRA with PEFT; target modules `q` and `v`; rank 8; alpha 16
- Epochs: 5; training steps: {trainer.state.global_step}
- Precision: FP32 (FP16/BF16 disabled)
- Final recorded training loss: {training_output.training_loss:.4f}
- Final validation loss: {validation_metrics.get('eval_loss', float('nan')):.4f}

## Parameter efficiency
| Measurement | Value |
|---|---:|
| Total parameters | {total_params:,} |
| Trainable parameters | {trainable_params:,} |
| Frozen parameters | {frozen_params:,} |
| Trainable percentage | {trainable_percentage:.2f}% |

## Evaluation
ROUGE-L F1 is an exploratory word-sequence overlap metric on one example from each of five held-out intents.

| Model | Mean ROUGE-L F1 |
|---|---:|
| Base | {metric_summary['mean_base_rougeL_f1']:.4f} |
| LoRA | {metric_summary['mean_lora_rougeL_f1']:.4f} |

This small test is not a factual-accuracy or semantic-similarity evaluation. See `evaluation/comparison_results.csv`, `evaluation/test_evaluation_metrics.csv`, and `evaluation/unseen_questions_results.csv` for outputs.

## Files
- `data/`: generated FAQ dataset and intent-separated train/validation/test CSVs
- `training_results/`: Trainer checkpoints and logs
- `college_faq_lora_adapter/`: saved LoRA adapter and tokenizer
- `evaluation/`: baseline comparison, unseen-question demo and ROUGE-L results
- `training_metrics.json`, `evaluation_summary.json`, `resource_info.json`: recorded metrics

## Reproduce
1. Open this notebook in Google Colab.
2. Select a GPU runtime if available.
3. Choose **Runtime → Run all** and authorize Google Drive.
4. Wait for training, evaluation and artifact checks to finish.

## Limitations and responsible use
- The data is synthetic and represents only 30 underlying FAQ intents.
- Some answers are generic, and the test set is very small.
- ROUGE-L is lexical overlap, not factual correctness.
- Verify every FAQ answer and URL against current official IEM sources before deployment. Do not use the model as the sole source for deadlines, fees, eligibility or other changing information.
