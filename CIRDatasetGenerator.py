import os
import json
import time
import openai
import re
import language_tool_python

class CIRDatasetGenerator:

    def __init__(self, output_dir: str, api_key_file: str, prompts: list[str], dataset_file: str):

        self.output_dir = output_dir
        self.api_key_file = api_key_file
        self.prompts = prompts
        self.dataset_file = dataset_file

        self.make_output_directory()
        self.api_key = self.load_api_key()
        self.dataset = self.load_dataset()

        self.stage_1_input_files = None
        self.stage_2_input_files = None
        self.stage_3_input_files = None

        self.stage_1_batches = None
        self.stage_2_batches = None
        self.stage_3_batches = None

    def make_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def load_api_key(self) -> str:
        try:
            with open(self.api_key_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to load API key: {e}")

    def load_dataset(self) -> dict:

        if not os.path.isfile(self.dataset_file):
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_file}")

        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
        
    def shard_output_directory(self):

        for index, entry in enumerate(self.dataset):
            batch_index = index // 1000  
            image_pair_index = index + 1     

            batch_id = f'{batch_index + 1:04d}'
            image_pair_id = f'{image_pair_index:06d}'

            batch_path = os.path.join(self.output_dir, batch_id)
            image_pair_path = os.path.join(batch_path, image_pair_id)
            
            os.makedirs(image_pair_path, exist_ok=True)
            
            reference_image_url = entry.get('reference_image')
            target_image_url = entry.get('target_image')

            data = {
                "batch_id": batch_id,
                "image_pair_id": image_pair_id,
                "query_image": { 
                    "url": reference_image_url,
                    "descriptors": None
                },
                "retrieved_image": {
                    "url": target_image_url,
                    "descriptors": None
                },
                "difference_captions": None
            }

            json_file_path = os.path.join(image_pair_path, f'output.json')

            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4) 

    def execute(self):
        self.shard_output_directory()
        self.run_stages()
        self.data_post_processing()

    def run_stages(self):
        for i in range(1, 4):
            self.create_batch_input_files(stage = i)
            self.collect_batch_input_files(stage = i)
            self.send_batches(stage = i)
            self.collect_responses(stage = i)
            print(f"completed: stage {i}")


    def create_batch_input_files(self, stage: int):
                
        for batch_id in os.listdir(self.output_dir):
            batch_path = os.path.join(self.output_dir, batch_id)
            
            if os.path.isdir(batch_path):
                entries = [] 
                
                for image_pair_id in os.listdir(batch_path):
                    image_pair_path = os.path.join(batch_path, image_pair_id)
                    
                    if os.path.isdir(image_pair_path):
                        output_file_path = os.path.join(image_pair_path, "output.json")
                        
                        if os.path.isfile(output_file_path):
                            with open(output_file_path, 'r') as file:
                                data = json.load(file)

                                if stage == 1:
                                    entry = {
                                        "custom_id": f"{data['batch_id']}-{data['image_pair_id']}",
                                        "method": "POST",
                                        "url": "/v1/chat/completions",
                                        "body": {
                                            "model": "gpt-4o",
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": self.prompts[stage - 1]
                                                },
                                                {
                                                    "role": "user",
                                                    "content": [
                                                        {
                                                        "type": "image_url",
                                                        "image_url": {
                                                            "url": data['query_image']['url']
                                                        }
                                                    }
                                                    ]
                                                }
                                            ],
                                            "max_tokens": 1500
                                        }
                                    }

                                elif stage == 2:
                                    entry = {
                                        "custom_id": f"{data['batch_id']}-{data['image_pair_id']}",
                                        "method": "POST",
                                        "url": "/v1/chat/completions",
                                        "body": {
                                            "model": "gpt-4o",
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": self.prompts[stage - 1]
                                                },
                                                {
                                                    "role": "user",
                                                    "content": [
                                                        {
                                                        "type": "image_url",
                                                        "image_url": {
                                                            "url": data['retrieved_image']['url']
                                                        }
                                                    }
                                                    ]
                                                },
                                                {
                                                    "role": "user",
                                                    "content": data['query_image']['descriptors']
                                                }
                                            ],
                                            "max_tokens": 1500
                                        }
                                    }

                                elif stage == 3:
                                    entry = {
                                        "custom_id": f"{data['batch_id']}-{data['image_pair_id']}",
                                        "method": "POST",
                                        "url": "/v1/chat/completions",
                                        "body": {
                                            "model": "gpt-4o",
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": self.prompts[stage - 1]
                                                },
                                                {
                                                    "role": "user",
                                                    "content": data['query_image']['descriptors']
                                                },
                                                {
                                                    "role": "user",
                                                    "content": data['retrieved_image']['descriptors']
                                                }
                                            ],
                                            "max_tokens": 1500
                                        }
                                    }
                                    
                            entries.append(entry)
                
                output_directory = os.path.join(batch_path, "input_files")
                os.makedirs(output_directory, exist_ok=True)

                jsonl_file_path = os.path.join(output_directory, f"stage_{stage}_input.jsonl")
                with open(jsonl_file_path, 'w') as jsonl_file:
                    for entry in entries:
                        jsonl_file.write(json.dumps(entry) + '\n')


    def collect_input_files(self, file_path: str, stage: int) -> list[str]:
        keyword = f"stage_{stage}"

        filenames = [os.path.join(file_path, filename) for filename in os.listdir(file_path) if filename.endswith('.jsonl') and keyword in filename]

        return filenames
        
    def collect_batch_input_files(self, stage: int) -> list[str]:

        input_files = []

        for batch_id in os.listdir(self.output_dir):
            batch_path = os.path.join(self.output_dir, batch_id)

            if os.path.isdir(batch_path):
                input_file_path = os.path.join(batch_path, "input_files")

                if os.path.isdir(input_file_path):
                    filenames = self.collect_input_files(input_file_path, stage)
                    input_files.extend(filenames)

        setattr(self, f"stage_{stage}_input_files", filenames)

    def send_batches(self, stage: int):

        openai.api_key = self.api_key
 
        batches = []

        input_files = getattr(self, f"stage_{stage}_input_files")

        for input_file in input_files:
            input_file_directory = os.path.dirname(input_file)
            batch_directory = os.path.dirname(input_file_directory)
            batch_id = os.path.basename(batch_directory)

            try:
                uploaded_file = openai.files.create(
                    file=open(input_file, "rb"),
                    purpose="batch"
                )

                batch = openai.batches.create(
                    input_file_id=uploaded_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": batch_id}
                )

                batches.append(batch)

            except Exception as e:
                print(f"Failed to create batch for {batch_id}: {e}")

        setattr(self, f"stage_{stage}_batches", batches)

    def collect_responses(self, stage: int):

        openai.api_key = self.api_key

        batch_files = getattr(self, f"stage_{stage}_batches")

        for batch in batch_files:
            updated_batch = openai.batches.retrieve(batch.id)
            
            while updated_batch.status not in ['completed', 'failed']:
                time.sleep(10)
                updated_batch = openai.batches.retrieve(batch.id)

            if updated_batch.status == 'completed':
                output_file_id = updated_batch.output_file_id
            
                file_response = openai.files.content(output_file_id)

                for json_object in file_response.text.splitlines():
                    try:
                        response = json.loads(json_object)
                        
                        content = response['response']['body']['choices'][0]['message']['content']
                        
                        custom_id = response['custom_id']

                        batch_id = custom_id.split('-')[0]
                        image_pair_id = custom_id.split('-')[1]
                
                        target_dir = os.path.join(self.output_dir, batch_id, image_pair_id)
                        output_path = os.path.join(target_dir, "output.json")
                        
                        with open(output_path, 'r') as file:
                            data = json.load(file)
                        
                        if stage == 1:
                            data['query_image']['descriptors'] = content
                        elif stage == 2:
                            data['retrieved_image']['descriptors'] = content
                        elif stage == 3:
                            data['difference_captions'] = content
                
                        with open(output_path, 'w') as file:
                            json.dump(data, file, indent=4)
                            
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON object: {e}")
                
            elif batch.status == 'failed':
                continue

    def clean(self, captions: str) -> list[str]:

        cleaned_sentences = []

        tool = language_tool_python.LanguageTool('en-US')

        for caption in str(captions).split('.'):
            caption = re.sub(r'^[^A-Z]*', '', caption)
            matches = tool.check(caption)
            if len(matches) == 0 and len(caption) > 0:
                cleaned_sentences.append(caption.strip()+'.')
            else:
                continue

        return cleaned_sentences

    def data_post_processing(self, min_caption_length=100):

        processed_data = []

        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as json_file:
                        try:
                            data = json.load(json_file)
                            if len(data['difference_captions']) > min_caption_length:
                                entry = {
                                    'query_image': data['query_image']['url'],
                                    'retrieved_image': data['retrieved_image']['url'],
                                    'difference_captions': self.clean(data['difference_captions'])
                                }
                                processed_data.append(entry)
                        except (json.JSONDecodeError, KeyError):
                            continue 

        with open('dataset.json', 'w') as outfile:
            json.dump(processed_data, outfile, indent=4)