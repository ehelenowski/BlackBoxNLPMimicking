data:
	python3 data_generator.py
labels: data
	python3 data_labeler.py
model: labels
	python3 new_model.py
interactive: model
	python3 predictor.py
