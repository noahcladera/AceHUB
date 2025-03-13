.PHONY: all install install-prod update download extract normalize feature-engineering train predict clip clean

# Install development dependencies from requirements.txt
install:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt

# Install production dependencies from requirements-prod.txt
install-prod:
	@echo "Installing production dependencies..."
	pip install -r requirements-prod.txt

# Update pipeline paths by running the path update script
update:
	@echo "Updating pipeline paths..."
	python lkxc.py

# Download and preprocess raw videos (e.g., using yt-dlp and FFmpeg)
download:
	@echo "Downloading and processing raw videos..."
	python src/data/acquisition.py

# Extract pose data with MediaPipe
extract:
	@echo "Extracting pose data from videos..."
	python src/data/pose_extraction.py

# Normalize the extracted pose data (both spatially and temporally)
normalize:
	@echo "Normalizing pose data..."
	python src/data/normalization.py

# Perform feature engineering and merge manual labels with normalized CSV data
feature-engineering:
	@echo "Performing feature engineering..."
	python src/data/feature_engineering.py

# Train the tennis stroke detection model (using, e.g., LSTM or Transformer)
train:
	@echo "Training the detection model..."
	python src/training/trainer.py

# Run inference on new videos to detect strokes
predict:
	@echo "Running inference..."
	python src/inference/predictor.py

# Generate video clips based on detected stroke segments
clip:
	@echo "Generating clips from detection results..."
	python src/inference/clip_generator.py

# The "all" target runs the full pipeline in sequence:
all: update download extract normalize feature-engineering train predict clip

# Clean up any temporary files or build artifacts if necessary.
clean:
	@echo "Cleaning up temporary files..."
	# For example, remove __pycache__ directories or other caches:
	find . -type d -name "__pycache__" -exec rm -r {} +