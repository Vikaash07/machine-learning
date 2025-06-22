# Brain Tumor Classification Using ResNet18

A deep learning approach for automated brain tumor classification using MRI scans, implemented with a custom ResNet18 architecture in PyTorch. This project demonstrates how to build and train a neural network for medical image classification using Google Colab.

## 🧠 Overview

This implementation leverages a custom ResNet18 architecture to classify brain tumors from MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor. The project combines both TensorFlow/Keras for data preprocessing and PyTorch for model implementation, providing a comprehensive solution for medical image classification.

## ✨ Features

- **Custom ResNet18 Implementation**: Built from scratch with BasicBlock residual components
- **Hybrid Framework Approach**: TensorFlow for data handling, PyTorch for model training
- **Google Colab Integration**: Designed for easy execution in cloud environments
- **Comprehensive Data Pipeline**: Automated dataset loading and preprocessing
- **Visual Training Monitoring**: Real-time loss and accuracy plotting during training
- **Model Persistence**: Automatic saving of best performing models
- **Prediction Pipeline**: Ready-to-use functions for inference on new images

## 🛠️ Technical Architecture

### Model Components
- **BasicBlock**: Residual learning blocks with skip connections
- **ResNet18**: 4-layer architecture with [2, 2, 2, 2] block configuration
- **Batch Normalization**: Integrated throughout the network
- **Adaptive Average Pooling**: For handling variable input sizes

### Data Processing
- **Image Size**: 224×224 pixels
- **Batch Size**: 16 images per batch
- **Normalization**: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
- **Classes**: 4 tumor categories

## 📋 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pillow>=8.3.0
```

## 🚀 Setup and Installation

### Google Colab Setup

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Install required packages (if needed):
```python
!pip install torch torchvision
```

### Dataset Structure

Organize your brain tumor dataset in Google Drive:
```
/content/drive/MyDrive/brain tumor data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 📁 File Structure

```
brain-tumor-classification/
├── brain_tumor_E0122049.ipynb    # Main notebook with complete implementation
├── README.md                     # This file
└── LICENSE                       # Project license
```

## 💻 Usage

### Running the Complete Pipeline

The main notebook `brain_tumor_E0122049.ipynb` contains the complete implementation. Key sections include:

1. **Data Loading and Exploration**:
```python
# Generate data paths and labels
train_data_dir = '/content/drive/MyDrive/brain tumor data/Training'
test_data_dir = '/content/drive/MyDrive/brain tumor data/Testing'
```

2. **Model Architecture**:
```python
# Initialize ResNet18 model
model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=4).to(device)
```

3. **Training**:
```python
# Train the model with visualization
train_and_save_model(model, criterion, optimizer, train_loader, valid_loader, 
                     num_epochs=15, model_path='best_model.pth')
```

4. **Prediction**:
```python
# Load trained model and make predictions
model = load_model('best_model.pth')
predicted_class = predict_image(model, image_path)
```

### Key Functions

**Training Function**:
- `train_and_save_model()`: Complete training loop with validation and plotting
- Automatically saves the best model based on validation accuracy
- Displays real-time training/validation loss and accuracy graphs

**Inference Functions**:
- `load_model()`: Loads a saved model for inference
- `predict_image()`: Predicts tumor class for a single image

## 📊 Model Configuration

### Training Parameters
- **Epochs**: 15
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Device**: CUDA (if available) or CPU

### Data Augmentation
- Image resizing to 224×224
- Tensor conversion
- Normalization with mean and std of 0.5

## 🔬 Model Architecture Details

```python
class BasicBlock(nn.Module):
    # Residual block with:
    # - 3x3 convolutions
    # - Batch normalization
    # - ReLU activation
    # - Skip connections

class ResNet18(nn.Module):
    # Complete ResNet18 implementation:
    # - Initial 7x7 conv + maxpool
    # - 4 residual layers
    # - Adaptive average pooling
    # - Fully connected classifier
```

## 📈 Training Monitoring

The training function provides real-time visualization of:
- Training and validation loss curves
- Training and validation accuracy curves
- Console output for each epoch's metrics

## 🎯 Prediction Pipeline

To classify a new brain MRI image:

```python
# Load the trained model
model = load_model('best_model.pth')

# Predict on new image
image_path = '/path/to/your/mri/image.jpg'
predicted_class = predict_image(model, image_path)
class_name = train_dataset.classes[predicted_class]
print(f'Predicted class: {class_name}')
```

## 🏥 Medical Image Classes

The model classifies brain MRI images into four categories:
1. **Glioma**: A type of brain tumor that originates in glial cells
2. **Meningioma**: Tumor arising from the meninges
3. **No Tumor**: Normal brain tissue without tumors
4. **Pituitary**: Tumor in the pituitary gland

## ⚠️ Important Notes

- **Google Colab Environment**: This project is specifically designed for Google Colab
- **GPU Acceleration**: Automatically uses CUDA if available in Colab
- **Data Path**: Update data paths according to your Google Drive structure
- **Model Saving**: Models are saved in the current working directory

## 🔮 Future Enhancements

- **Data Augmentation**: Advanced augmentation techniques for better generalization
- **Cross-Validation**: K-fold cross-validation for robust model evaluation
- **Transfer Learning**: Pre-trained weights from ImageNet or medical datasets
- **3D Analysis**: Extension to volumetric MRI data
- **Web Interface**: Deployment as a web application
- **DICOM Support**: Direct processing of medical imaging formats

## 🏥 Clinical Disclaimer

**Important**: This model is designed for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in the Jupyter notebook
4. Test thoroughly in Google Colab
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Technical References

- **ResNet Paper**: "Deep Residual Learning for Image Recognition" by He et al.
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Medical Image Analysis**: Various papers on CNN applications in medical imaging

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- TensorFlow/Keras for data preprocessing utilities
- Google Colab for providing free GPU resources
- Medical imaging community for dataset contributions

## 📧 Contact

**Author**: Vikaash07
- GitHub: [@Vikaash07](https://github.com/Vikaash07)

---

**Development Environment**: Google Colab  
**Last Updated**: 34 minutes ago  
**Project ID**: E0122049
