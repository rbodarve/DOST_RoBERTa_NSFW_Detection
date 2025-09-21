# DOST RoBERTa NSFW Text Detection

A comprehensive Jupyter notebook implementation for fine-tuning the DOST RoBERTa-tl sentiment analysis model for NSFW (Not Safe For Work) text detection. This project provides an end-to-end pipeline from dataset preparation to mobile-ready model export with ONNX and TensorFlow Lite formats.

## Features

- **Filipino Language Support**: Uses DOST-ASTI RoBERTa-tl sentiment analysis model
- **Multi-class Classification**: Positive, Neutral, and NSFW text detection
- **Automatic Dataset Generation**: Creates sample dataset if none provided
- **Multiple Export Formats**: PyTorch, ONNX, and TensorFlow Lite models
- **Mobile Optimization**: Quantized models for Android/iOS deployment
- **Comprehensive Evaluation**: Detailed metrics and performance analysis
- **Google Colab Ready**: Designed for seamless cloud execution

## Model Information

**Base Model**: `dost-asti/RoBERTa-tl-sentiment-analysis`
- Pre-trained RoBERTa model for Tagalog sentiment analysis
- Developed by DOST-ASTI (Department of Science and Technology - Advanced Science and Technology Institute)
- Fine-tuned for Filipino/Tagalog text processing

## System Requirements

- Python 3.8 or higher
- PyTorch with CUDA support (recommended)
- Transformers library 4.x
- TensorFlow 2.x (for TFLite export)
- At least 4GB RAM, 2GB GPU memory recommended

## Installation

The notebook automatically handles dependency installation:

```python
required_packages = [
    'transformers[torch]',
    'datasets', 
    'torch',
    'pandas',
    'scikit-learn',
    'onnx',
    'onnxruntime',
    'optimum[onnxruntime]',
    'tensorflow',
]
```

## Usage

### Basic Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook dost_robert.ipynb
```

Or execute it directly:

```bash
python dost_robert.ipynb
```

For Google Colab, upload the notebook and run all cells sequentially.

### Dataset Requirements

#### Expected Format
The model expects a CSV file named `dataset.csv` with the following structure:

```csv
text,label
"Magandang umaga sa lahat",0
"Kumusta kayo",1
"inappropriate content here",2
```

#### Label Encoding
- `0`: Positive/Safe content
- `1`: Neutral content  
- `2`: NSFW/Inappropriate content

#### Automatic Dataset Generation
If no dataset is provided, the notebook creates a sample dataset with:
- 25 positive examples ("Good job 1", "Good job 2", etc.)
- 20 neutral examples ("hello", "world", "computer", etc.)
- 25 NSFW examples (placeholder "inappropriate1", etc.)

## Project Structure

```
project/
├── dost_robert.ipynb          # Main notebook
├── dataset.csv                # Training dataset (optional)
├── models/
│   ├── roberta_tl_nsfw/      # Trained model directory
│   ├── nsfw_model.onnx       # ONNX export
│   └── nsfw_model.tflite     # TensorFlow Lite export
└── metrics/
    └── metrics.txt            # Training metrics and evaluation
```

## Key Components

### Dataset Loading and Preparation
```python
df = load_dataset('dataset.csv')
X_train, X_val, y_train, y_val = split_dataset(df)
```
- Validates CSV structure and labels (0, 1, 2)
- Handles missing data and type conversion
- Stratified train/validation split (80/20)

### Model Configuration
```python
MODEL_NAME = "dost-asti/RoBERTa-tl-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={0: "Positive", 1: "Neutral", 2: "NSFW"},
    label2id={"Positive": 0, "Neutral": 1, "NSFW": 2}
)
```

### Training Parameters
- **Epochs**: 5 (configurable)
- **Batch Size**: 16 per device
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 128 tokens
- **Optimization**: AdamW with 0.01 weight decay
- **Early Stopping**: Based on evaluation loss

### Tokenization and Preprocessing
```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Max length: 128 tokens (suitable for word/phrase detection)
# Padding: Dynamic (handled by data collator)
# Truncation: Enabled
```

## Export Capabilities

### ONNX Export
```python
export_to_onnx(model_dir, "models/nsfw_model.onnx")
```
- Opset version 14 for compatibility
- Dynamic batch size support
- Optimized for inference

### TensorFlow Lite Export  
```python
export_to_tflite_from_pt(model_dir, "models/nsfw_model.tflite")
```
- Direct PyTorch to TensorFlow conversion
- DEFAULT optimization applied
- Mobile-ready quantized model

## Model Performance

### Evaluation Metrics
The notebook provides comprehensive evaluation including:
- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, Recall, F1-score per class
- **Confusion Matrix**: Detailed prediction analysis
- **Training Loss Curves**: Loss progression over epochs

### Expected Performance
On sample dataset:
- Training typically converges within 3-5 epochs
- Validation accuracy varies based on dataset quality
- Model size: ~500MB (PyTorch), ~125MB (ONNX), ~125MB (TFLite)

## Inference and Testing

### Built-in Testing
```python
test_inference("models/roberta_tl_nsfw", ["sample text", "another example"])
```

### Classification Logic
The model adapts sentiment analysis for NSFW detection:
- **Class 0 (Positive)** → Mapped to "SAFE"
- **Class 1 (Neutral)** → Mapped to "SAFE" 
- **Class 2 (Negative)** → Mapped to "NSFW"

### Sample Output
```
Text: 'hello world' -> SAFE (confidence: 0.8234)
  Class probabilities: Negative=0.0123, Neutral=0.1643, Positive=0.8234
```

## Mobile Deployment

### Android Integration
```java
// Load ONNX model with ONNX Runtime
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession("nsfw_model.onnx");
```

### iOS Integration
```swift
// Load TensorFlow Lite model
guard let interpreter = try? Interpreter(modelPath: "nsfw_model.tflite") else {
    return
}
```

### Model Specifications
- **Input**: Tokenized text (input_ids, attention_mask)
- **Output**: 3-class logits [Positive, Neutral, NSFW]
- **Preprocessing**: RoBERTa tokenization with 128 max length

## Error Handling and Troubleshooting

### Common Issues

1. **Dataset Not Found**: Automatically creates sample dataset
2. **CUDA Out of Memory**: Reduce batch size from 16 to 8
3. **Transformers Version**: Handles both old and new TrainingArguments API
4. **Export Failures**: Graceful fallback with detailed error messages

### Dependency Management
The notebook includes automatic dependency checking and installation with fallback options for missing packages.

### Version Compatibility
- **Transformers**: Supports both old (`evaluation_strategy`) and new (`eval_strategy`) API
- **PyTorch**: Compatible with 1.8+ and 2.x versions
- **TensorFlow**: Requires 2.x for TFLite conversion

## Advanced Configuration

### Custom Training Parameters
Modify the training configuration in the `train_model` function:

```python
training_args_dict = {
    'num_train_epochs': 10,        # Increase epochs
    'per_device_train_batch_size': 8,  # Reduce batch size
    'learning_rate': 1e-5,         # Lower learning rate
    'weight_decay': 0.02,          # Increase regularization
}
```

### Custom Dataset Integration
1. Prepare CSV file with 'text' and 'label' columns
2. Ensure labels are integers (0, 1, 2)
3. Place file as `dataset.csv` in project directory
4. Run the notebook - it will automatically detect and use your data

## Performance Optimization

### GPU Utilization
- Automatic CUDA detection and usage
- Batch size optimization for available memory
- Mixed precision training support (if available)

### Memory Management
- Dynamic padding to reduce memory usage
- Gradient accumulation for large effective batch sizes
- Model checkpointing to prevent data loss

## Contributing

Areas for contribution:
- Enhanced preprocessing for Filipino text
- Additional export formats (CoreML, TensorRT)
- Improved sample dataset with real examples
- Performance benchmarking on different hardware

## License

This project uses the DOST-ASTI RoBERTa-tl model. Please refer to their licensing terms and ensure compliance with usage policies.

## Ethical Considerations

This tool is designed for content moderation and safety applications. Users should:
- Ensure training data is ethically sourced
- Test thoroughly before production deployment
- Consider bias and fairness in classification decisions
- Respect privacy and data protection regulations

## Support

For issues and questions:
1. Check the automatic error handling in the notebook
2. Verify dataset format and label encoding
3. Ensure all dependencies are properly installed
4. Review model-specific documentation from DOST-ASTI

## Changelog

- **v1.0**: Initial implementation with DOST RoBERTa-tl
- **v1.1**: Added automatic dataset generation
- **v1.2**: Enhanced export capabilities (ONNX + TFLite)
- **v1.3**: Improved error handling and compatibility