from CIRDatasetGenerator import CIRDatasetGenerator

def main():
    output_dir = '...' 
    api_key_file = '...'
    dataset_file = '...' 
    prompts = [
        "...", 
        "...",
        "..."
    ]

    generator = CIRDatasetGenerator(
        output_dir=output_dir,
        api_key_file=api_key_file,
        prompts=prompts,
        dataset_file=dataset_file
    )

    generator.execute()

if __name__ == '__main__':
    main()