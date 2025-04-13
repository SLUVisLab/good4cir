from good4cir.src.CIRDatasetGenerator import CIRDatasetGenerator

def main():
    
    output_dir = '...' 
    api_key_file = '...'
    dataset_file = '...'

    stage_1 = """
                Curate a list of up to X objects in the image from most
                prominent to least prominent. For each object, generate a
                list of descriptors. The descriptors should describe the exact
                appearance of the object, mentioning any fine-grained details.
                
                Example: Object Name: [“object description 1”, “object
                description 2”, . . . , “object description N”]

                Format objects and descriptors as a JSON output.
            """
    stage_2 = """
                Here is an image and a list of descriptors that describe a
                different image. Curate a similar list for this image by doing
                the following:

                1. If there is a new object in this image that isn’t described
                in the description of the other image, generate a new set
                of descriptors.
                2. If the description of an object from the other image
                matches the appearance of an object in this image, use
                the exact same list of descriptors.
                3. If the object appears different in this image in comparison
                to the description from the other image, generate a new
                set of descriptors.

                Format objects and descriptors as a JSON output.
            """
    stage_3 = """
                The following are two sets of objects with descriptors that
                describe two different images that have been determined to
                be different in some ways. Analyze both lists and generate
                short and comprehensive instructions on how to modify the
                first image to look more like the second image. Be sure to
                mention what objects have been added, removed, or modified.
                Don’t mention “Image 1” and “Image 2” or any similar
                phrasing. Focus on having variety in the styles of captions
                that are generated, and make sure they mimic human-like
                syntactical structure and diction. Format all difference 
                captions as a list of sentences ending in periods.
            """

    prompts = [
        stage_1,
        stage_2,
        stage_3
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