import os
import pathlib

def create_structure():
    # Define the root directory name
    root_dir = "tennis-stroke-detection"
    
    # Define the structure as nested dictionaries
    # Empty string means it's a directory, otherwise it's a file with that content
    structure = {
        "config": {
            "data_config.yaml": "",
            "model_config.yaml": "",
            "pipeline_config.yaml": ""
        },
        "data": {
            "raw": {},
            "interim": {
                "pose_data": {},
                "normalized": {}
            },
            "processed": {
                "features": {},
                "labels": {}
            },
            "external": {}
        },
        "models": {
            "base.py": "",
            "sequence": {
                "lstm.py": "",
                "transformer.py": ""
            },
            "feature_extractors": {
                "pose_encoder.py": ""
            },
            "ensemble": {
                "stacking.py": ""
            }
        },
        "src": {
            "data": {
                "acquisition.py": "",
                "pose_extraction.py": "",
                "normalization.py": "",
                "feature_engineering.py": "",
                "dataset.py": ""
            },
            "training": {
                "trainer.py": "",
                "metrics.py": "",
                "callbacks.py": ""
            },
            "inference": {
                "predictor.py": "",
                "clip_generator.py": ""
            },
            "visualization": {
                "pose_visualizer.py": "",
                "results_visualizer.py": ""
            }
        },
        "notebooks": {
            "exploratory": {},
            "modeling": {},
            "evaluation": {}
        },
        "tests": {
            "unit": {},
            "integration": {},
            "fixtures": {}
        },
        "experiments": {
            "runs": {},
            "artifacts": {}
        },
        "abilities": {
            "manual_labeling_ability.py": "",
            "data_validator.py": "",
            "model_converter.py": ""
        },
        "scripts": {
            "legacy": {}
        },
        "docs": {
            "api": {},
            "data_pipeline.md": "",
            "model_architecture.md": ""
        }
    }
    
    # Root level files
    root_files = {
        ".env.example": "",
        ".gitignore": "",
        "pyproject.toml": "",
        "setup.py": "",
        "requirements.txt": "",
        "requirements-prod.txt": "",
        "Makefile": "",
        "README.md": "# Tennis Stroke Detection\n\nAI-powered tennis stroke detection and analysis."
    }

    def create_nested_structure(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            if isinstance(content, dict):
                # It's a directory
                os.makedirs(path, exist_ok=True)
                create_nested_structure(path, content)
            else:
                # It's a file
                with open(path, 'w') as f:
                    f.write(content)

    # Create the root directory
    os.makedirs(root_dir, exist_ok=True)
    
    # Create the nested structure
    create_nested_structure(root_dir, structure)
    
    # Create root level files
    for filename, content in root_files.items():
        with open(os.path.join(root_dir, filename), 'w') as f:
            f.write(content)

    print(f"Project structure created successfully in '{root_dir}' directory!")

if __name__ == "__main__":
    create_structure()