# Image_Caption_Generator-
Image Caption Generator using Deep Learning on Flickr8K dataset
# Image Caption Generator using CNN + LSTM (Flickr8k Dataset)

This project generates human-like captions for images using a deep learning model. It combines a Convolutional Neural Network (CNN) for extracting image features and a Recurrent Neural Network (RNN) with LSTM cells for generating captions based on those features.

---

##  Project Overview

- Goal: Automatically generate a relevant caption for a given image.
- **Model Architecture**: Encoder-Decoder model with:
  - Encoder: CNN (InceptionV3)
  - Decoder: LSTM
- **Dataset**: Flickr8k – contains 8,000 images, each with 5 human-written captions.

---

##  Technologies Used

- Python 
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- NLTK (for BLEU score)
- Pretrained InceptionV3 (as encoder)

---

##  Model Architecture

- **Encoder (CNN)**: InceptionV3 pretrained on ImageNet. It extracts a 2048-d feature vector from input images.
- **Decoder (LSTM)**:
  - Takes the CNN feature and caption input
  - Generates the next word at each time step
- Text Preprocessing:
  - Tokenization
  - Padding
  - Word Embeddings

---

## Caption Generation Techniques

| Method        | Description |
|---------------|-------------|
| Greedy Search | Picks the word with the highest probability at each step |
| Beam Search   | Explores multiple possible caption paths and picks the best-scoring one |

---

## Evaluation

- BLEU-2 and BLEU-4 scores were used to evaluate the quality of generated captions.
- BLEU compares the generated caption to one or more reference captions.

---

##  Dataset

- Flickr8k: [https://github.com/jbrownlee/Datasets](https://github.com/jbrownlee/Datasets)
- Contains 8,000 images, each with 5 human-written captions.
- We used:
  - `Flickr8k.token.txt` for captions
  - `Flickr8k_Dataset/` for image files

---

##  How to Run

1. Clone this repo
   ```bash
   git clone https://github.com/yourusername/image-caption-generator.git
   cd image-caption-generator
2. Install dependencies
- pip install -r requirements.txt
  Prepare the dataset
  3. Place images in Flickr8k_Dataset/

4. Run the notebook
  Open Image_Caption_Generator.ipynb in Jupyter Notebook or Colab

##  Sample Output
<img width="710" height="679" alt="image" src="https://github.com/user-attachments/assets/6d41dc81-b527-47fd-ad3d-275520bfc229" />

# To improve caption accuracy, we can:
- Use a larger dataset
- Include color/gender attributes as part of supervised training 
  
