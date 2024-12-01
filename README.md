# DesignLLM

DesignLLM focuses on customers interacting with our model to create customized designs with ease.

Highlights:

-**Template-Driven Language Model**: Powered by a proprietary corpus with structured templates, enabling accurate and consistent design generation tailored to your requirements.

-**Dynamic Prompt Adaptation**: The model leverages predefined templates and adjusts them based on user feedback to ensure personalized and optimized outputs.

-**Cost-Effective Realism**: Designs are affordable and practical, striking a perfect balance between creativity and usability.

-**Versatile Deployment**: Fully adaptable for use on web platforms, mobile applications, and even text-based systems.

A sample:

<img src="./test.png" alt="drawing" width="200"/>

## Repository Structure

The repository is organized as follows:

- **InstructionsGenerated/**: Private generated instructions or guidelines related to the project.

- **Text2Img/**: Includes scripts and resources for converting text descriptions into images.

- **llm_finetune_scripts/**: Scripts for fine-tuning language scripts:
- 
  -- .yaml files: config for [torchtune](https://github.com/pytorch/torchtune)
  
  -- instr_xx.json files: the dataset for finetuning
  
  -- ds_config.json: config for [deepspeed](https://github.com/microsoft/DeepSpeed)
  
  -- logging.txt: The final utilized model training logging.
  
- **public_prompts/**: Prompts collected publically.

- **Commands.sh**: Include all the steps to use our model.

- **gpu_test.sh**: A shell script designed to test GPU configurations and performance.





