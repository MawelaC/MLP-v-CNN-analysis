# MLP vs CNN: A Complexity and Efficiency Comparison

This project investigates how Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) compare on image classification tasks of increasing complexity, and how each architecture uses its parameters under a fixed parameter budget.

The analysis is split into two parts:

- **Part A - Complexity Ladder**: Compares standard MLP and CNN architectures on MNIST (low complexity) and CIFAR-10 (high complexity) to identify where MLP performance begins to degrade relative to CNNs.
- **Part B - Fixed Parameter Budget**: Constrains both architectures to a similar total parameter count (~475,000–525,000) on CIFAR-10 to study how architectural design affects parameter efficiency, generalization, and training time.

## Contents

- `MLP_vs_CNN.ipynb` - Main notebook containing implementation, training, results, and analysis.

## Part A: Complexity Ladder

1. **Implementation** - Custom `MultiLayerPerceptron` and `ConvNN` (built from a reusable `ConvBlock` of Conv2d → BatchNorm2d → ReLU) classes implemented in PyTorch.
2. **Results Table** - Test accuracy of both models on MNIST and CIFAR-10.
3. **Training & Validation Curves** - Loss and accuracy curves plotted against percentage of training completed (to account for early stopping producing different epoch counts per model).
4. **Analysis** - Examines why the MLP/CNN performance gap is minimal on MNIST but substantial on CIFAR-10.
5. **Reflection** - Discussion of inductive bias, spatial structure, and why MLPs struggle as image complexity increases.

**Key finding**: On MNIST, the MLP (~98%) performs close to the CNN (~99.1%), since the task has low spatial complexity. On CIFAR-10, the gap widens significantly (MLP ~51% validation accuracy vs CNN ~78–80%), since CNNs exploit spatial locality and translation invariance that MLPs cannot.

## Part B: Fixed Parameter Budget

1. **Architecture Report** - A parameter-matched CNN (`AdaptedConvNN`) and MLP are built and compared layer-by-layer using a custom parameter-counting utility and `torchinfo.summary`.
2. **Base Model Training** - Both models trained on CIFAR-10 with matched hyperparameters (batch size, learning rate, patience, minimum epochs).
3. **Hyperparameter Tuning** - Grid search over learning rate, batch size, and dropout for each architecture.
4. **Efficiency Analysis** - Compares test accuracy vs training time and examines how each architecture allocates its parameter budget.
5. **Reflection** - Proposes a locally-connected (non-weight-shared) MLP variant as a middle ground between full dense connectivity and convolution.

**Key finding**: At a matched parameter budget, the CNN reaches ~75% test accuracy versus the MLP's ~54%, despite a longer training time (~748s vs ~612s). This is attributed to the CNN concentrating parameters in feature extraction (with weight sharing) rather than spreading them across unstructured dense connections.

## Requirements

- Python 3
- PyTorch (`torch`, `torchvision`)
- `numpy`, `pandas`, `matplotlib`
- `torchinfo`

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib torchinfo
```

## Datasets

The notebook automatically downloads:

- **MNIST** (28×28 greyscale digits)
- **CIFAR-10** (32×32 RGB images, 10 classes)

via `torchvision.datasets`. Note: dataset paths in the notebook default to a Kaggle working directory (`/kaggle/working/data`) - update the `root` argument in `load_data()` if running outside Kaggle.

## Usage

1. Open `MLP_vs_CNN.ipynb` in Jupyter or Kaggle.
2. Run all cells sequentially - Part A trains the baseline models, Part B builds and tunes the parameter-matched models.
3. Results tables and training curves are generated inline.

## Author

Chegofatjo Mawela
