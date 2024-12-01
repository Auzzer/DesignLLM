import pandas as pd
import random
import glob

def load_templates(template_file):
    with open(template_file, "r") as file:
        templates = file.readlines()
    return [template.strip() for template in templates]

# Construct input using a random template
def construct_input(style, feature, detail, decor, atmosphere, templates):
    template = random.choice(templates)
    return template.format(style=style, feature=feature, detail=detail, decor=decor, atmosphere=atmosphere)

# Process multiple CSV files to create a dataset
def process_csv_files(input_files, output_file, template_file):
    # Load templates from the file
    templates = load_templates(template_file)
    data = []

    for file in input_files:
        df = pd.read_csv(file, delimiter=",", skip_blank_lines=True, encoding="utf-8")
        for _, row in df.iterrows():
            # Construct input using a random template
            input_text = construct_input(
                style=row["Style"],
                feature=row["Feature"],
                detail=row["Detail"],
                decor=row["Decor"],
                atmosphere=row["Atmosphere"],
                templates=templates
            )
            # Use the prompt as the output
            output_text = row["Prompt"]
            data.append({"input": input_text, "output": output_text})

    # Save the dataset
    dataset = pd.DataFrame(data)
    dataset.to_csv(output_file, index=False)

# Specify input and output paths
input_files = glob.glob("./origin/bedroom_design_prompts_*.csv")
output_file = "./llama_training_dataset_with_templates.csv"
template_file = "./templates.txt"

# Process the files
process_csv_files(input_files, output_file, template_file)

print(f"Dataset saved to {output_file}")
