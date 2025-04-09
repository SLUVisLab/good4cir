from good4cir.CIRDatasetGenerator import CIRDatasetGenerator

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

    generator.shard_output_directory()

    for i in range(1, 4):
        generator.create_batch_input_files(stage = i)
        generator.collect_batch_input_files(stage = i)
        generator.send_batches(stage = i)
        generator.collect_responses(stage = i)

if __name__ == '__main__':
    main()