# SecIDS-CNN: Advanced Convolutional Neural Network for Intrusion Detection in Cybersecurity and Automotive Applications


### Model Description

SecIDS-CNN is a high-performance Convolutional Neural Network (CNN) model developed specifically for Intrusion Detection Systems (IDS) in cybersecurity and automotive network applications. Leveraging temporal patterns in network traffic, SecIDS-CNN identifies and classifies malicious activity with high accuracy, designed to meet the real-time security demands of vehicular and automotive networks. This model supports proactive threat mitigation, helping to protect in-vehicle and connected systems against cyber threats that could impact operational safety.

- **Developed by:** Keyvan Hardani
- **Model Type:** Convolutional Neural Network (CNN) for Intrusion Detection
- **Languages:** English, German
- **License:** Creative Commons Attribution Non Commercial 4.0 (cc-by-nc-4.0)
- **Finetuned from model:** None

### Model Sources

- **Repository:** https://huggingface.co/Keyven/SecIDS-CNN
### For access to the SecIDS-CNN model, visit [our Hugging Face page](https://huggingface.co/Keyven/SecIDS-CNN) and request access.


## Uses

### Direct Use

SecIDS-CNN can be directly deployed for real-time intrusion detection within cybersecurity monitoring systems. Its design supports seamless integration into automotive communication networks, enabling anomaly detection within complex, connected vehicular systems.

### Downstream Use

Potential applications include broader network monitoring platforms and integrated security systems in automotive and connected vehicle environments.

### Out-of-Scope Use

SecIDS-CNN is not suited for non-network data or applications outside the network security and automotive domains. Misuse may include attempts to deploy it in systems without real-time requirements or in unrelated cybersecurity needs.

## Bias, Risks, and Limitations

SecIDS-CNN, while highly accurate, may have a minor bias toward benign traffic when optimized for recall, which could lead to rare false negatives. Additionally, its effectiveness depends on access to live network data, essential for real-time intrusion detection.

### Recommendations

Users should be aware of the model’s optimal use cases in real-time network environments and its limitations in handling unrelated or non-automotive network types.


## How to Get Started with SecIDS-CNN

To get started with SecIDS-CNN, you can import the model and use it in your Python project. Follow the steps below:

### Step 1: Install Dependencies

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/keyvanhardani/SecIDS-CNN.git
cd SecIDS-CNN
pip install -r requirements.txt
```

### Step 2: Import the Model

Once dependencies are installed, you can import the model into your Python project:

```python
from secids_cnn import SecIDSModel
```

### Step 3: Load and Use the Model

To evaluate SecIDS-CNN’s real-time detection on sample network traffic data:

```python
# Initialize the model
model = SecIDSModel()

# Load your network traffic data (example)
data = load_network_data('path/to/your/data.csv')

# Make predictions
predictions = model.predict(data)

# Output results
print("Intrusion Detection Results:", predictions)
```

This setup allows you to test SecIDS-CNN on provided sample data or integrate it into larger projects for real-time intrusion detection.

## Training Details

### Training Data

The dataset for SecIDS-CNN consists of labeled network traffic, distinguishing between benign and malicious activity. It includes data from general network and automotive sources, with features capturing packet flows, timing, and network behavior.

### Training Procedure

The model’s training pipeline encompasses data preprocessing, feature extraction, and training on temporal network data patterns.

#### Training Hyperparameters

- **Precision Type:** FP32
- **Batch Size:** 32
- **Epochs:** 50

### Compute Requirements

SecIDS-CNN was trained on a multi-GPU setup, with optimizations for real-time performance in security-critical applications.

## Evaluation

### Testing Data and Metrics

#### Testing Data

The model was evaluated on a balanced set of benign and malicious network traffic records, sourced from both general cybersecurity and automotive domains.

#### Metrics

SecIDS-CNN’s evaluation included accuracy, precision, recall, F1-score, ROC curve, and AUC, chosen for their relevance to classification performance in security applications.

### Results

- **Accuracy:** 97.72%
- **Precision:** 97.74%
- **Recall:** 97.72%
- **F1-Score:** 0.9772

SecIDS-CNN demonstrated high reliability, achieving almost 98% accuracy in intrusion detection and benign traffic classification.

## Model Examination

Feature importance was analyzed using SHAP (SHapley Additive exPlanations) to gain insight into feature contributions. This interpretability measure supports transparency and offers guidance for refining the model for intrusion detection.

- **Top Features:** Packet_Length_Mean, Flow_Duration
- **Least Impactful Features:** Bwd_Packet_Length_Mean, Idle_Mean

## Environmental Impact

The estimated carbon footprint for training SecIDS-CNN was calculated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute).

- **Hardware:** Multi-GPU setup (NVIDIA RTX 4070, RTX 4090Ti)
- **Training Duration:** 

  Batch Size: 32
  Epochs: 50
  Training Duration: ~72 hours on RTX 4090Ti
  Emissions: ~15 kg CO₂

## Technical Specifications

### Model Architecture

SecIDS-CNN utilizes a multi-layer convolutional architecture, optimized for high-throughput analysis of network traffic data, with an emphasis on capturing time-based patterns.

### Compute Infrastructure

- **Software:** TensorFlow, Python, Keras

### Supported Hardware

This model is lightweight and versatile for inference across a wide range of hardware, including:

- **CPUs**: Compatible with standard CPUs, allowing easy deployment on nearly any system.
- **GPUs**: Optimized for all GPUs (primarily used for training), but also enables faster inference if needed.
- **Microcontrollers and Edge Devices**: With a small model size (~700 KB), it supports microprocessors and edge devices, such as Raspberry Pi, NVIDIA Jetson Nano, and other embedded systems.

This compatibility ensures flexibility for various applications in automotive and cybersecurity environments.

## Citation

**BibTeX:**

```bibtex
@misc{secids-cnn,
  author = {Keyvan Hardani},
  title = {SecIDS-CNN: Advanced Convolutional Neural Network for Intrusion Detection},
  year = {2023},
  note = {Available under CC BY-NC 4.0}
}

@misc {keyvan_hardani_2024,
	author       = { {Keyvan Hardani} },
	title        = { SecIDS-CNN (Revision 5daf4a4) },
	year         = 2024,
	url          = { https://huggingface.co/Keyven/SecIDS-CNN },
	doi          = { 10.57967/hf/3351 },
	publisher    = { Hugging Face }
}
