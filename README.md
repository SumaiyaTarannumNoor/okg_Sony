# OKG: On-the-fly Keyword Generation in Search Sponsered Advertising

## Description

This is the official implementation for the paper **OKG: On-the-fly Keyword Generation in Search Sponsered Advertising** authored by Zhao Wang , Briti Gangopadhyay , Mengjie Zhao , Shingo Takamatsu,  submitted to COLING 2025. OKG is a dynamic framework leveraging LLM agent to adaptively generate keywords for sponsored search advertising. Additionally, we provided the first publicly accessible dataset with real ad keyword data, offering a valuable resource for future research in keyword optimization. 

![Example Figure](./architecture.png)

## Installation

To install and run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/wang1946may7/OKG.git
   ```

2.  Creating a New Environment from a YAML File: You can create a new Conda environment from a YAML file on any system with Conda installed using:
   ```bash
   conda env create -f environment.yml
```


## Configuration File (.ini)

To run the agent, you need to AT LEAST replace items listed as follows including the keys and the ad product name. 
The project uses an `.ini` file to store various configurations and hyperparameters. Here’s a breakdown of the sections and their purposes:


### [CAMPAIGN]
- `PRODUCT_NAME`: The name of the product being advertised.

### [KEY] (need to replace them with your own keys)
- `OPENAI_GPT4_API_KEY`: API key for accessing GPT-4.
- `OPENAI_EMBEDDING_API_KEY`: API key for accessing OpenAI embeddings.
- `OPENAI_GPT4_AZURE_OPENAI_ENDPOINT`: Endpoint for GPT-4 Azure integration.
- `OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT`: Endpoint for embedding API Azure integration.
- `SERPAPI_API_KEY`: API key for accessing SERPAPI services.

## Dataset

We present a publicly accessible dataset that includes real Japanese keyword data with its KPIs across various domains. The [dataset](https://github.com/wang1946may7/OKG/tree/main/dataset) includes real advertisement deliveries for 10 Sony products and IT services: Sony electronic devices like cameras and TVs, Sony financial services including Sony Bank mortgages and health insurance, and Sony AI platforms such as the Sony Neural Network Console and Prediction One. The dataset contains not only the actual delivered keywords but also the performance of each keyword, including search volume, clicks, competitor score, and cost-per-click.

[Dataset](https://github.com/wang1946may7/OKG/tree/main/dataset) - See this folder for more details on our datasets. Accessed on October 1, 2024.

## Usage
Run the main script:
   ```bash
   python main.py
   ```
## License
This project is licensed under the [Attribution-NonCommercial 4.0 International Lisence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## Contact
For any questions or issues, feel free to reach out: Zhao.Wang@sony.com or this github repo for any information.

## Cite
If you use or reference OKG, please cite us with the following BibTeX entry:
```bibtex
@inproceedings{wang-etal-2025-okg,
    title = "{OKG}: On-the-Fly Keyword Generation in Sponsored Search Advertising",
    author = "Wang, Zhao  and
      Gangopadhyay, Briti  and
      Zhao, Mengjie  and
      Takamatsu, Shingo",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics: Industry Track",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    pages = "115--127"
}
