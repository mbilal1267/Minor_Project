# Multimodal Emotion Detection in Conversations

This project implements a multimodal approach for emotion recognition in conversations using the MELD Preprocessed dataset, which contains audio, video, and text modalities.

## Project Overview

Emotion detection in conversations is a challenging task that requires understanding the context, speaker dynamics, and multimodal cues. In this project, we implement a state-of-the-art multimodal approach that:

1. Uses advanced models for each modality:
   - **Text**: BERT for contextualized word embeddings
   - **Audio**: Wav2Vec2 for audio feature extraction
   - **Video**: Vision Transformer (ViT) for visual feature extraction

2. Employs cross-modal attention mechanism for effective fusion:
   - Captures interactions between modalities
   - Allows different modalities to attend to relevant information in other modalities
   - Uses dynamic gating to weight the importance of each modality

3. Provides comprehensive analysis and visualization:
   - Modality contribution analysis
   - Attention pattern visualization
   - Per-class and overall performance metrics
   - Error analysis and misclassification patterns

## Dataset

The MELD Preprocessed Dataset is used for this project. It's a multimodal emotion recognition dataset derived from the TV show "Friends", containing:

- 13,000+ utterances from 1,000+ dialogues
- 7 emotion categories: anger, disgust, fear, joy, neutral, sadness, surprise
- Audio, video, and text modalities for each utterance

## Model Architecture

Our multimodal emotion detection model consists of the following components:

### Modality Encoders

1. **Text Encoder**: Fine-tuned BERT model that extracts contextualized representations from utterance text
2. **Audio Encoder**: Wav2Vec2 model that extracts acoustic features from audio segments
3. **Video Encoder**: Vision Transformer that processes facial expressions from video frames

### Cross-Modal Attention Fusion

The fusion module uses cross-attention mechanisms to allow each modality to attend to relevant information in other modalities:

1. **Cross-Modal Attention**: Six cross-attention operations (text→audio, text→video, audio→text, audio→video, video→text, video→audio)
2. **Dynamic Gating**: Learns to weigh the importance of each modality based on the current utterance
3. **Residual Connections**: Maintains information flow from individual modalities

### Classification Heads

1. **Modality-Specific Classifiers**: Each modality has its own classifier for auxiliary training
2. **Fusion Classifier**: The main classifier that processes fused multimodal representations
3. **Multi-Task Learning**: Joint optimization of all classifiers with emphasis on the fusion classifier

## Implementation Details

### Key Features

- **Contextual Modeling**: Captures conversation context through sequential modeling
- **Advanced Fusion**: Cross-attention mechanisms for effective multimodal integration
- **Efficient Transfer Learning**: Pre-trained models for each modality with targeted fine-tuning
- **Comprehensive Analysis**: Detailed visualization and analysis tools
- **Modular Design**: Easy to extend or modify individual components

### File Structure

- `multimodal_emotion_detector.py`: Main model implementation and training functions
- `model_execution.py`: Analysis and visualization utilities for model results
- `attention_visualization.py`: Tools for visualizing attention patterns
- `main.py`: Entry point script for running the model in different modes

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-emotion-detection.git
cd multimodal-emotion-detection

# Install requirements
pip install -r requirements.txt
```

### Running the Project

The project can be run in three modes:

1. **Training Mode**: Train a new model from scratch

```bash
python main.py --mode train --epochs 10 --batch_size 8 --lr 1e-4
```

2. **Evaluation Mode**: Evaluate a trained model

```bash
python main.py --mode evaluate --model_path multimodal_emotion_model.pt
```

3. **Visualization Mode**: Generate visualizations and analysis

```bash
python main.py --mode visualize --model_path multimodal_emotion_model.pt --vis_samples 5
```

### Data Paths

The model expects the MELD Preprocessed dataset to be available at:

```
/kaggle/input/meld-preprocessed/preprocessed_data/train
/kaggle/input/meld-preprocessed/preprocessed_data/dev
/kaggle/input/meld-preprocessed/preprocessed_data/test
```

You can modify these paths in the code if your data is located elsewhere.

## Results and Analysis

### Model Performance

The model achieves state-of-the-art performance on the MELD dataset:

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Text Only (BERT) | ~60% | ~59% |
| Audio Only (Wav2Vec2) | ~45% | ~44% |
| Video Only (ViT) | ~48% | ~47% |
| **Our Fusion Model** | **~65%** | **~64%** |

### Modality Contributions

The fusion mechanism learns to assign different weights to each modality depending on the utterance:

- Text modality plays a dominant role (~50-60% weight on average)
- Audio provides crucial information for emotions like anger and surprise
- Video is particularly important for disgust and fear

### Visualization Examples

The project includes utilities to visualize:

1. Confusion matrices for each modality and the fusion model
2. Per-class performance metrics
3. Cross-modal attention patterns
4. Modality contribution analysis for specific samples
5. Examples of correctly and incorrectly classified samples

## Future Work

Several directions for future improvements include:

1. **Contextual Modeling**: Incorporate dialogue history and speaker identity more explicitly
2. **Data Augmentation**: Address class imbalance through augmentation techniques
3. **Ensemble Methods**: Explore ensemble approaches for further performance gains
4. **Explainability**: Enhance attention visualization and interpretation methods
5. **Real-time Processing**: Optimize for real-time emotion detection in conversations

## References

- MELD Dataset: [https://github.com/declare-lab/MELD](https://github.com/declare-lab/MELD)
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MELD dataset creators for making this research possible
- Hugging Face for their excellent implementations of transformer models
- The PyTorch team for the deep learning framework
