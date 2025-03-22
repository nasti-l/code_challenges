from kfp.v2 import compiler
from kfp.v2.dsl import component, pipeline, Output, Dataset
from google.cloud import aiplatform

@component
def preprocess_data(data: str, output_data: Output[Dataset]):
    with open(output_data.path, 'w') as f:
        f.write(data.upper())

@component
def inference(data: Dataset):
    with open(data.path, 'r') as f:
        content = f.read()
    print("Inference Result:", content[::-1]) 

@pipeline(name="simple-pipeline", pipeline_root="gs://your-bucket-name/pipeline-root")
def simple_pipeline(input_data: str):
    processed = preprocess_data(data=input_data)
    inference(data=processed.output)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=simple_pipeline,
        package_path="simple_pipeline.json"
    )
    aiplatform.PipelineJob(
    display_name="simple-pipeline",
    template_path="simple_pipeline.json",
    pipeline_root="gs://example_bucket_vertex/pipeline-root",
    parameter_values={"input_data": "hello world"}).run()
